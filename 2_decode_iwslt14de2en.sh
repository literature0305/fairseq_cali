#!/usr/bin/env bash

# evaluation
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 --beam 20 --remove-bpe --calibration-mode None        
# Decoding config (calibration modes)

# 'None'
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
