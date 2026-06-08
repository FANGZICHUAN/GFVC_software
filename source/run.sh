#!/bin/bash

if [ $2 == "encoder" ]; then
python $2.py  --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14} --closed_loop_encoding ${15} --use_base_layer ${16} --bl_qp ${17} --device ${18}
elif [ $2 == "decoder" ]; then
python $2.py --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14} --closed_loop_encoding ${15} --use_base_layer ${16} --bl_qp ${17}   --encoder_type ${18} --color_calibration ${19} --device ${20}
fi