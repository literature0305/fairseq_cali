# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)


logger = logging.getLogger(__name__)


class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        type_calibration=None,
        use_pseudo_conf=False,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        if type_calibration is not None and ('_conf' in type_calibration) and (self.training or use_pseudo_conf):
            ###### 2.0 inference
            encoder_out = self.encoder(
                src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
            )
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=features_only,
                alignment_layer=alignment_layer,
                alignment_heads=alignment_heads,
                src_lengths=src_lengths,
                return_all_hiddens=return_all_hiddens,
            ) # prev_output_tokens torch.Size([56, 69]),  # decoder_out torch.Size([56, 69, 6632])

            ###### 4.0 print option
            if torch.randperm(10000)[0] == 0:
                prev_output_tokens_backup = prev_output_tokens.detach().clone()
                print_option=True
            else:
                print_option=False

            ###### 5.0 stochastically use true prev-output
            if torch.randperm(10)[0] < 5:
                use_true_label=True
            else:
                use_true_label=False

            if use_true_label:
                prev_output_tokens_one_hot = torch.nn.functional.one_hot(prev_output_tokens, num_classes=decoder_out[0].size(-1))
                confidence = torch.ones(prev_output_tokens.size()).to(decoder_out[0].device).to(decoder_out[0].dtype)

                # conf: torch.Size([56, 69])
                # conf_tmp: torch.Size([56, 68, 6632])
                # prev torch.Size([56, 69, 6632])
                # decoder_out[0] torch.Size([56, 69, 6632])
                conf_tmp = (prev_output_tokens_one_hot[:,1:] * torch.softmax(decoder_out[0], dim=-1)[:,:-1]).max(-1).values

                confidence[:,1:] = conf_tmp.detach()

                if print_option:
                    print('(use_true_label) conf mean:', confidence.mean()) # Error here
                    print('(use_true_label) conf:', confidence)
                    print('(use_true_label) prev argmax:', torch.argmax(prev_output_tokens_one_hot[:,1:], dim=-1))
                    print('(use_true_label) decoder out argmax:', torch.argmax(torch.softmax(decoder_out[0], dim=-1)[:,:-1], dim=-1))
            else:
                ###### 5.0 get prev_output_tokens with post-processing
                mask_1 = (prev_output_tokens==1).to(prev_output_tokens.device).to(prev_output_tokens.dtype)
                mask_2 = (prev_output_tokens==2).to(prev_output_tokens.device).to(prev_output_tokens.dtype)
                prev_output_tokens = torch.zeros(prev_output_tokens.size()).to(prev_output_tokens.device).to(prev_output_tokens.dtype)
                prev_output_tokens[:,1:] = torch.argmax(decoder_out[0].detach(), dim=-1)[:,:-1]

                prev_output_tokens = prev_output_tokens * (1-mask_1) + mask_1
                prev_output_tokens = prev_output_tokens * (1-mask_2) + mask_2 * 2

                ###### 6.0 get confidence
                decoder_out_max = torch.softmax(decoder_out[0], dim=-1).detach().max(-1).values # B,T,v -> B,T
                confidence = torch.ones(decoder_out_max.size()).to(decoder_out_max.device).to(decoder_out_max.dtype)
                confidence[:,1:] = decoder_out_max[:,:-1]

                assert confidence.size() == prev_output_tokens.size(), 'conf: {}, prev: {}'.format(confidence.size(), prev_output_tokens.size())

                ###### 7.0 print
                if print_option:
                    acc = (prev_output_tokens_backup == prev_output_tokens).sum() / torch.ones(prev_output_tokens_backup.size()).sum()
                    conf = confidence.mean()
                    print('acc wo teacher forcing:', acc)
                    print('conf wo teacher forcing:', conf)
        else:
            confidence=None
        
        ###################################################################### do calibration ######################################################################
        # i: modules/transformer_layer.TransformerEncoderLayerBase
        for i in self.encoder.layers:
            i.turn_calibration_mode_encoder(type_calibration)
        for i in self.decoder.layers:
            i.turn_calibration_mode_decoder(type_calibration)
            i.set_confidence_decoder(confidence)
        self.decoder.turn_calibration_mode_decoder_temp(type_calibration)
        self.decoder.set_confidence_decoder_temp(confidence)
        #############################################################################################################################################################
        
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
