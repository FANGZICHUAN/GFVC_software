#!/bin/bash

if [ $2 == "encoder" ]; then
python $2_scalable.py  --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14}  --use_base_layer ${15} --bl_qp ${16} --SRLR_feature_size ${17} --scalable_base_model_name ${18} --scalable_feature_code_mode ${19} --scalable_feature_diff_mode ${20}

elif [ $2 == "decoder" ]; then
python $2_scalable.py --model_name $1 --original_seq $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --ae_qf ${10} --ae_qf ${11} --sei_coordinate_qf ${12} --sei_matrix_qf ${13} --no_analysis_in_decoder ${14}  --use_base_layer ${15} --bl_qp ${16}   --SRLR_feature_size ${17} --scalable_base_model_name ${18}  --scalable_feature_code_mode ${19} --scalable_feature_diff_mode ${20} --encoder_type ${21}
fi