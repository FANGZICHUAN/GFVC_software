#!/bin/bash

if [ $2 == "encoder" ]; then
python $2.py  --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14}  --use_base_layer ${15} --bl_qp ${16} --device ${17} --smart_keyframe ${18} --num_keyframes ${19}
elif [ $2 == "decoder" ]; then
python $2.py --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14}  --use_base_layer ${15} --bl_qp ${16}   --encoder_type ${17} --color_calibration ${18} --device ${19}
fi