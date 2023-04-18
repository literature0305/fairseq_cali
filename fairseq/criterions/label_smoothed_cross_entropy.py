# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from omegaconf import II

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, optimizer=None, sample_valid=None, stage_validation=False, reduce=True, update_num=None):
        update_num_threshold_for_calibration_func_training=-1 #30000
        if update_num is None:
            update_num = update_num_threshold_for_calibration_func_training + 1

        if stage_validation:
            # validation stage
            ####################### without calibration #######################
            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            if update_num > update_num_threshold_for_calibration_func_training:
                ####################### with temperature-scaling #######################
                type_calibration='temperature'
                net_output_valid_temp = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_temp, nll_loss_valid_temp = self.compute_loss(model, net_output_valid_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_temp"] = loss_valid_temp.data
                logging_output["nll_loss_valid_temp"] = nll_loss_valid_temp.data

                if self.report_accuracy:
                    n_correct_valid_temp, total_valid_temp = self.compute_accuracy(model, net_output_valid_temp, sample)
                    logging_output["n_correct_temp"] = utils.item(n_correct_valid_temp.data)
                    logging_output["total_temp"] = utils.item(total_valid_temp.data)

                ####################### with att-calibration #######################
                type_calibration='att_temp'
                net_output_valid_att = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_att, nll_loss_valid_att = self.compute_loss(model, net_output_valid_att, sample, reduce=reduce)
                
                logging_output["loss_valid_att"] = loss_valid_att.data
                logging_output["nll_loss_valid_att"] = nll_loss_valid_att.data

                if self.report_accuracy:
                    n_correct_valid_att, total_valid_att = self.compute_accuracy(model, net_output_valid_att, sample)
                    logging_output["n_correct_att"] = utils.item(n_correct_valid_att.data)
                    logging_output["total_att"] = utils.item(total_valid_att.data)

                ####################### with mh-att-calibration #######################
                type_calibration='mh_att_temp'
                net_output_valid_mh_att = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_mh_att, nll_loss_valid_mh_att = self.compute_loss(model, net_output_valid_mh_att, sample, reduce=reduce)
                
                logging_output["loss_valid_mh_att"] = loss_valid_mh_att.data
                logging_output["nll_loss_valid_mh_att"] = nll_loss_valid_mh_att.data

                if self.report_accuracy:
                    n_correct_valid_mh_att, total_valid_mh_att = self.compute_accuracy(model, net_output_valid_mh_att, sample)
                    logging_output["n_correct_mh_att"] = utils.item(n_correct_valid_mh_att.data)
                    logging_output["total_mh_att"] = utils.item(total_valid_mh_att.data)

                ####################### with adaptive att-calibration #######################
                type_calibration='ad_att_temp'
                net_output_valid_ad_att = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_ad_att, nll_loss_valid_ad_att = self.compute_loss(model, net_output_valid_ad_att, sample, reduce=reduce)
                
                logging_output["loss_valid_ad_att"] = loss_valid_ad_att.data
                logging_output["nll_loss_valid_ad_att"] = nll_loss_valid_ad_att.data

                if self.report_accuracy:
                    n_correct_valid_ad_att, total_valid_ad_att = self.compute_accuracy(model, net_output_valid_ad_att, sample)
                    logging_output["n_correct_ad_att"] = utils.item(n_correct_valid_ad_att.data)
                    logging_output["total_ad_att"] = utils.item(total_valid_ad_att.data)

                ####################### with multi head adaptive att-calibration #######################
                type_calibration='mh_ad_att_temp'
                net_output_valid_mh_ad_att = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_mh_ad_att, nll_loss_valid_mh_ad_att = self.compute_loss(model, net_output_valid_mh_ad_att, sample, reduce=reduce)
                
                logging_output["loss_valid_mh_ad_att"] = loss_valid_mh_ad_att.data
                logging_output["nll_loss_valid_mh_ad_att"] = nll_loss_valid_mh_ad_att.data

                if self.report_accuracy:
                    n_correct_valid_mh_ad_att, total_valid_mh_ad_att = self.compute_accuracy(model, net_output_valid_mh_ad_att, sample)
                    logging_output["n_correct_mh_ad_att"] = utils.item(n_correct_valid_mh_ad_att.data)
                    logging_output["total_mh_ad_att"] = utils.item(total_valid_mh_ad_att.data)

                ####################### without calibration wo teacher forcing #######################
                type_calibration='None_wo_tf'
                net_output_valid_wo_tf = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf, nll_loss_valid_wo_tf = self.compute_loss(model, net_output_valid_wo_tf, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf"] = loss_valid_wo_tf.data
                logging_output["nll_loss_valid_wo_tf"] = nll_loss_valid_wo_tf.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf, total_valid_wo_tf = self.compute_accuracy(model, net_output_valid_wo_tf, sample)
                    logging_output["n_correct_wo_tf"] = utils.item(n_correct_valid_wo_tf.data)
                    logging_output["total_wo_tf"] = utils.item(total_valid_wo_tf.data)

                ####################### with temperature scaling wo teacher forcing #######################
                type_calibration='temperature_wo_tf'
                net_output_valid_wo_tf_temperature = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf_temp, nll_loss_valid_wo_tf_temp = self.compute_loss(model, net_output_valid_wo_tf_temperature, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf_temp"] = loss_valid_wo_tf_temp.data
                logging_output["nll_loss_valid_wo_tf_temp"] = nll_loss_valid_wo_tf_temp.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf_temp, total_valid_wo_tf_temp = self.compute_accuracy(model, net_output_valid_wo_tf_temperature, sample)
                    logging_output["n_correct_wo_tf_temp"] = utils.item(n_correct_valid_wo_tf_temp.data)
                    logging_output["total_wo_tf_temp"] = utils.item(total_valid_wo_tf_temp.data)

                ####################### with att-temperature scaling wo teacher forcing #######################
                type_calibration='att_temp_wo_tf'
                net_output_valid_wo_tf_att_temp = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf_att_temp, nll_loss_valid_wo_tf_att_temp = self.compute_loss(model, net_output_valid_wo_tf_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf_att_temp"] = loss_valid_wo_tf_att_temp.data
                logging_output["nll_loss_valid_wo_tf_att_temp"] = nll_loss_valid_wo_tf_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf_att_temp, total_valid_wo_tf_att_temp = self.compute_accuracy(model, net_output_valid_wo_tf_att_temp, sample)
                    logging_output["n_correct_wo_tf_att_temp"] = utils.item(n_correct_valid_wo_tf_att_temp.data)
                    logging_output["total_wo_tf_att_temp"] = utils.item(total_valid_wo_tf_att_temp.data)

                ####################### with mh-att-temperature scaling wo teacher forcing #######################
                type_calibration='mh_att_temp_wo_tf'
                net_output_valid_wo_tf_mh_att_temp = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf_mh_att_temp, nll_loss_valid_wo_tf_mh_att_temp = self.compute_loss(model, net_output_valid_wo_tf_mh_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf_mh_att_temp"] = loss_valid_wo_tf_mh_att_temp.data
                logging_output["nll_loss_valid_wo_tf_mh_att_temp"] = nll_loss_valid_wo_tf_mh_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf_mh_att_temp, total_valid_wo_tf_mh_att_temp = self.compute_accuracy(model, net_output_valid_wo_tf_mh_att_temp, sample)
                    logging_output["n_correct_wo_tf_mh_att_temp"] = utils.item(n_correct_valid_wo_tf_mh_att_temp.data)
                    logging_output["total_wo_tf_mh_att_temp"] = utils.item(total_valid_wo_tf_mh_att_temp.data)

                ####################### with ad-att-temperature scaling wo teacher forcing #######################
                type_calibration='ad_att_temp_wo_tf'
                net_output_valid_wo_tf_ad_att_temp = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf_ad_att_temp, nll_loss_valid_wo_tf_ad_att_temp = self.compute_loss(model, net_output_valid_wo_tf_ad_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf_ad_att_temp"] = loss_valid_wo_tf_ad_att_temp.data
                logging_output["nll_loss_valid_wo_tf_ad_att_temp"] = nll_loss_valid_wo_tf_ad_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf_ad_att_temp, total_valid_wo_tf_ad_att_temp = self.compute_accuracy(model, net_output_valid_wo_tf_ad_att_temp, sample)
                    logging_output["n_correct_wo_tf_ad_att_temp"] = utils.item(n_correct_valid_wo_tf_ad_att_temp.data)
                    logging_output["total_wo_tf_ad_att_temp"] = utils.item(total_valid_wo_tf_ad_att_temp.data)

                ####################### with mh-ad-att-temperature scaling wo teacher forcing #######################
                type_calibration='mh_ad_att_temp_wo_tf'
                net_output_valid_wo_tf_mh_ad_att_temp = model(**sample["net_input"], type_calibration=type_calibration)
                loss_valid_wo_tf_mh_ad_att_temp, nll_loss_valid_wo_tf_mh_ad_att_temp = self.compute_loss(model, net_output_valid_wo_tf_mh_ad_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_wo_tf_mh_ad_att_temp"] = loss_valid_wo_tf_mh_ad_att_temp.data
                logging_output["nll_loss_valid_wo_tf_mh_ad_att_temp"] = nll_loss_valid_wo_tf_mh_ad_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_wo_tf_mh_ad_att_temp, total_valid_wo_tf_mh_ad_att_temp = self.compute_accuracy(model, net_output_valid_wo_tf_mh_ad_att_temp, sample)
                    logging_output["n_correct_wo_tf_mh_ad_att_temp"] = utils.item(n_correct_valid_wo_tf_mh_ad_att_temp.data)
                    logging_output["total_wo_tf_mh_ad_att_temp"] = utils.item(total_valid_wo_tf_mh_ad_att_temp.data)

                ####################### with temperature scaling with conf #######################
                type_calibration='temperature_conf'
                net_output_valid_conf_temp = model(**sample["net_input"], type_calibration=type_calibration, use_pseudo_conf=True)
                loss_valid_conf_temp, nll_loss_valid_conf_temp = self.compute_loss(model, net_output_valid_conf_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_conf_temp"] = loss_valid_conf_temp.data
                logging_output["nll_loss_valid_conf_temp"] = nll_loss_valid_conf_temp.data

                if self.report_accuracy:
                    n_correct_valid_conf_temp, total_valid_conf_temp = self.compute_accuracy(model, net_output_valid_conf_temp, sample)
                    logging_output["n_correct_conf_temp"] = utils.item(n_correct_valid_conf_temp.data)
                    logging_output["total_conf_temp"] = utils.item(total_valid_conf_temp.data)

                ####################### with att-temperature scaling with conf #######################
                type_calibration='att_temp_conf'
                net_output_valid_conf_att_temp = model(**sample["net_input"], type_calibration=type_calibration, use_pseudo_conf=True)
                loss_valid_conf_att_temp, nll_loss_valid_conf_att_temp = self.compute_loss(model, net_output_valid_conf_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_conf_att_temp"] = loss_valid_conf_att_temp.data
                logging_output["nll_loss_valid_conf_att_temp"] = nll_loss_valid_conf_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_conf_att_temp, total_valid_conf_att_temp = self.compute_accuracy(model, net_output_valid_conf_att_temp, sample)
                    logging_output["n_correct_conf_att_temp"] = utils.item(n_correct_valid_conf_att_temp.data)
                    logging_output["total_conf_att_temp"] = utils.item(total_valid_conf_att_temp.data)

                ####################### with att-temperature scaling with conf #######################
                type_calibration='mh_att_temp_conf'
                net_output_valid_conf_mh_att_temp = model(**sample["net_input"], type_calibration=type_calibration, use_pseudo_conf=True)
                loss_valid_conf_mh_att_temp, nll_loss_valid_conf_mh_att_temp = self.compute_loss(model, net_output_valid_conf_mh_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_conf_mh_att_temp"] = loss_valid_conf_mh_att_temp.data
                logging_output["nll_loss_valid_conf_mh_att_temp"] = nll_loss_valid_conf_mh_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_conf_mh_att_temp, total_valid_conf_mh_att_temp = self.compute_accuracy(model, net_output_valid_conf_mh_att_temp, sample)
                    logging_output["n_correct_conf_mh_att_temp"] = utils.item(n_correct_valid_conf_mh_att_temp.data)
                    logging_output["total_conf_mh_att_temp"] = utils.item(total_valid_conf_mh_att_temp.data)

                ####################### with att-temperature scaling with conf #######################
                type_calibration='ad_att_temp_conf'
                net_output_valid_conf_ad_att_temp = model(**sample["net_input"], type_calibration=type_calibration, use_pseudo_conf=True)
                loss_valid_conf_ad_att_temp, nll_loss_valid_conf_ad_att_temp = self.compute_loss(model, net_output_valid_conf_ad_att_temp, sample, reduce=reduce)
                
                logging_output["loss_valid_conf_ad_att_temp"] = loss_valid_conf_ad_att_temp.data
                logging_output["nll_loss_valid_conf_ad_att_temp"] = nll_loss_valid_conf_ad_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_conf_ad_att_temp, total_valid_conf_ad_att_temp = self.compute_accuracy(model, net_output_valid_conf_ad_att_temp, sample)
                    logging_output["n_correct_conf_ad_att_temp"] = utils.item(n_correct_valid_conf_ad_att_temp.data)
                    logging_output["total_conf_ad_att_temp"] = utils.item(total_valid_conf_ad_att_temp.data)

                ####################### with att-temperature scaling with conf #######################
                type_calibration='mh_ad_att_temp_conf'
                net_output_valid_conf_mh_ad_att__temp = model(**sample["net_input"], type_calibration=type_calibration, use_pseudo_conf=True)
                loss_valid_conf_mh_ad_att_temp, nll_loss_valid_conf_mh_ad_att_temp = self.compute_loss(model, net_output_valid_conf_mh_ad_att__temp, sample, reduce=reduce)
                
                logging_output["loss_valid_conf_mh_ad_att_temp"] = loss_valid_conf_mh_ad_att_temp.data
                logging_output["nll_loss_valid_conf_mh_ad_att_temp"] = nll_loss_valid_conf_mh_ad_att_temp.data

                if self.report_accuracy:
                    n_correct_valid_conf_mh_ad_att__temp, total_valid_conf_mh_ad_att__temp = self.compute_accuracy(model, net_output_valid_conf_mh_ad_att__temp, sample)
                    logging_output["n_correct_conf_mh_ad_att__temp"] = utils.item(n_correct_valid_conf_mh_ad_att__temp.data)
                    logging_output["total_conf_mh_ad_att__temp"] = utils.item(total_valid_conf_mh_ad_att__temp.data)

                if torch.randperm(300)[0] == 0:
                    print('loss_valid_none:', loss, 'loss_valid_temp:', loss_valid_temp,'loss_valid_att:', loss_valid_att, 'loss_valid_mh_att:', loss_valid_mh_att, 'loss_valid_ad_att:', loss_valid_ad_att, 'loss_valid_mh_ad_att', loss_valid_mh_ad_att)
                    print('loss_valid_none_wo_tf:', loss_valid_wo_tf, 'loss_valid_temp_wo_tf:', loss_valid_wo_tf_temp,'loss_valid_att_wo_tf:', loss_valid_wo_tf_att_temp, 'loss_valid_mh_att_wo_tf:', loss_valid_wo_tf_mh_att_temp, 'loss_valid_ad_att_wo_tf:', loss_valid_wo_tf_ad_att_temp, 'loss_valid_mh_ad_att_wo_tf', loss_valid_wo_tf_mh_ad_att_temp)
                    print('loss_valid_conf_temp:', loss_valid_conf_temp,'loss_valid_conf_att_temp:', loss_valid_conf_att_temp, 'loss_valid_conf_mh_att_temp:', loss_valid_conf_mh_att_temp, 'loss_valid_conf_ad_att_temp:', loss_valid_conf_ad_att_temp, 'loss_valid_conf_mh_ad_att_temp', loss_valid_conf_mh_ad_att_temp)
            return loss, sample_size, logging_output
        else:
            ######################## stop gradient descent for encoder & decoder ########################
            for name, param in model.named_parameters(): 
                if 'scaling_factor' in name:
                    param.requires_grad=False
                else:
                    param.requires_grad=True
            ######################## stop gradient descent for encoder & decoder ########################

            net_output = model(**sample["net_input"])
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            if self.report_accuracy:
                n_correct, total = self.compute_accuracy(model, net_output, sample)
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)

            if optimizer is not None:
                with torch.autograd.profiler.record_function("backward"):
                    optimizer.backward(loss)

            if sample_valid is not None and update_num > update_num_threshold_for_calibration_func_training:
                type_calibration = 'temperature'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_for_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_temp = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_temp, _ = self.compute_loss(model, net_output_valid_temp, sample_valid, reduce=reduce)
                # TypeError: tuple indices must be integers or slices, not str
                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_temp)

                type_calibration = 'att_temp'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    if 'scaling_factor_for_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_att = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_att, _ = self.compute_loss(model, net_output_valid_att, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_att)

                type_calibration = 'mh_att_temp'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_mh_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_att = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_att, _ = self.compute_loss(model, net_output_valid_mh_att, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_att)

                type_calibration = 'ad_att_temp'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_ad_att = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_ad_att, _ = self.compute_loss(model, net_output_valid_ad_att, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_ad_att)

                type_calibration = 'mh_ad_att_temp'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_mh_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_ad_att = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_ad_att, _ = self.compute_loss(model, net_output_valid_mh_ad_att, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_ad_att)

                type_calibration = 'temperature_wo_tf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_wo_tf_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_temp_wo_tf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_temp_wo_tf, _ = self.compute_loss(model, net_output_valid_temp_wo_tf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_temp_wo_tf)

                type_calibration = 'att_temp_wo_tf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_wo_tf_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_att_temp_wo_tf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_att_temp_wo_tf, _ = self.compute_loss(model, net_output_valid_att_temp_wo_tf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_att_temp_wo_tf)

                type_calibration = 'mh_att_temp_wo_tf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_wo_tf_mh_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_att_temp_wo_tf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_att_temp_wo_tf, _ = self.compute_loss(model, net_output_valid_mh_att_temp_wo_tf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_att_temp_wo_tf)

                type_calibration = 'ad_att_temp_wo_tf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_wo_tf_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_ad_att_temp_wo_tf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_ad_att_temp_wo_tf, _ = self.compute_loss(model, net_output_valid_ad_att_temp_wo_tf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_ad_att_temp_wo_tf)

                type_calibration = 'mh_ad_att_temp_wo_tf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_wo_tf_mh_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_ad_att_temp_wo_tf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_ad_att_temp_wo_tf, _ = self.compute_loss(model, net_output_valid_mh_ad_att_temp_wo_tf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_ad_att_temp_wo_tf)

                type_calibration = 'temperature_conf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_conf_temp' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_temp_conf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_temp_conf, _ = self.compute_loss(model, net_output_valid_temp_conf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_temp_conf)

                type_calibration = 'att_temp_conf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_conf_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_att_temp_conf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_att_temp_conf, _ = self.compute_loss(model, net_output_valid_att_temp_conf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_att_temp_conf)

                type_calibration = 'mh_att_temp_conf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_conf_mh_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_att_temp_conf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_att_temp_conf, _ = self.compute_loss(model, net_output_valid_mh_att_temp_conf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_att_temp_conf)

                type_calibration = 'ad_att_temp_conf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_conf_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_ad_att_temp_conf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_ad_att_temp_conf, _ = self.compute_loss(model, net_output_valid_ad_att_temp_conf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_ad_att_temp_conf)

                type_calibration = 'mh_ad_att_temp_conf'
                ######################## stop gradient descent for encoder & decoder ########################
                for name, param in model.named_parameters(): 
                    # print('name:', name)
                    if 'scaling_factor_for_conf_mh_ad_att' in name:
                        param.requires_grad=True
                    else:
                        param.requires_grad=False
                    if 'encoder_attn.scaling_factor_for' in name:
                        param.requires_grad=False
                ######################## stop gradient descent for encoder & decoder ########################
                # for validation set
                net_output_valid_mh_ad_att_temp_conf = model(**sample_valid["net_input"], type_calibration=type_calibration)
                loss_valid_mh_ad_att_temp_conf, _ = self.compute_loss(model, net_output_valid_mh_ad_att_temp_conf, sample_valid, reduce=reduce)

                if optimizer is not None:
                    with torch.autograd.profiler.record_function("backward"):
                        optimizer.backward(loss_valid_mh_ad_att_temp_conf)

                ### print ###
                if torch.randperm(5000)[0] == 0:
                    print('(training) loss:', loss, 'loss_temp:', loss_valid_temp, 'loss_att:', loss_valid_att, 'loss_mh-att:', loss_valid_mh_att, 'loss_ad-att', loss_valid_ad_att, 'loss_mh-ad-att', loss_valid_mh_ad_att)
                    print('(training) loss_valid_temp_wo_tf:', loss_valid_temp_wo_tf, 'loss_valid_att-temp_wo_tf:', loss_valid_att_temp_wo_tf, 'loss_valid_mh-att_temp_wo_tf', loss_valid_mh_att_temp_wo_tf, 'loss_valid_ad_att_temp_wo_tf', loss_valid_ad_att_temp_wo_tf, 'loss_valid_mh_ad_att_temp_wo_tf', loss_valid_mh_ad_att_temp_wo_tf)
                    print('(training) loss_valid_temp_conf:', loss_valid_temp_conf, 'loss_valid_att_temp_conf:', loss_valid_att_temp_conf, 'loss_valid_mh_att_temp_conf', loss_valid_mh_att_temp_conf, 'loss_valid_ad_att_temp_conf', loss_valid_ad_att_temp_conf, 'loss_valid_mh_ad_att_temp_conf', loss_valid_mh_ad_att_temp_conf)
                    for name, param in model.named_parameters():
                        if 'scaling_factor' in name:
                            print('name:', name, 'param:', param)

            return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
