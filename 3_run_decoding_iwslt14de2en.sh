#!/usr/bin/env bash

# Set the number of times to run the script and the arguments to pass
# arguments=(None temperature att_temp mh_att_temp ad_att_temp mh_ad_att_temp)
arguments=(None temperature tau_temperature att_temp mh_att_temp d_att_temp d_plus_att_temp ad_att_temp d_ad_att_temp tau_att_temp mh_tau_att_temp d_tau_att_temp d_ad_tau_att_temp)

N=${#arguments[@]}
# arguments=(att_temp_conf mh_att_temp_conf ad_att_temp_conf mh_ad_att_temp_conf)

# Define the function that will run the script recursively
function run_recursive {
    # Get the number of arguments
    num_args=$1
    shift
    to_write=decode_400beam_${1}
    # If there are no more arguments, exit the function
    if [[ $num_args -eq 0 ]]; then
        return
    fi
    # Run the script with the next argument
    fairseq-generate data-bin/iwslt14.tokenized.de-en \
        --path checkpoints/checkpoint_best.pt \
        --batch-size 32 --beam 20 --remove-bpe --calibration-mode "$1" &> $to_write

    # Recursively call this function with the remaining arguments
    run_recursive "$((num_args - 1))" "${@:2}"
}

# Call the function to run the script recursively with all arguments
run_recursive $N "${arguments[@]}"



# # evaluation
# fairseq-generate data-bin/iwslt14.tokenized.de-en \
#     --path checkpoints/checkpoint_best.pt \
#     --batch-size 32 --beam 20 --remove-bpe --calibration-mode mh_att_temp_conf
    
#     # --batch-size 128 --beam 20 --remove-bpe
# for 
# # Decoding config (calibration modes)
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
