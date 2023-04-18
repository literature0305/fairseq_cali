#!/usr/bin/env bash

# evaluation
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 4 --beam 100 --remove-bpe --calibration-mode att_temp_conf
    
    # --batch-size 128 --beam 20 --remove-bpe

# Decoding config (calibration modes)
# 'temperature'
# 'att_temp'
# 'mh_att_temp'
# 'ad_att_temp'
# 'mh_ad_att_temp'

# 'temperature_wo_tf'
# 'att_temp_wo_tf'
# 'mh_att_temp_wo_tf'
# 'ad_att_temp_wo_tf'
# 'mh_ad_att_temp_wo_tf'

# 'temperature_conf'
# 'att_temp_conf'
# 'mh_att_temp_conf'
# 'ad_att_temp_conf'
# 'mh_ad_att_temp_conf'