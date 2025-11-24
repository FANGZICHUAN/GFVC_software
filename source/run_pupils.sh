#!/bin/bash

if [ $1 == "encoder_pupils" ]; then
python $1.py --original_seq $2 --gfvc $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --quantization_factor ${10} --adaptive ${11}
elif [ $1 == "decoder_pupils" ]; then
python $1.py --original_seq $2 --gfvc $3 --encoding_frames $4   --iframe_qp $5 --iframe_format $6 --seq_width $7 --seq_height $8 --feature_coding_mode $9 --quantization_factor ${10} --adaptive ${11}
fi