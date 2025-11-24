#!/bin/sh

./image_codecs/vtm/bin/EncoderAppStatic_VTM22_2_SEI -c ./image_codecs/vtm/cfg/encoder_lowdelay_vtm.cfg -c ./image_codecs/vtm/cfg/per-sequence/inputyuv420.cfg -c $5 -q $2 -i $1.yuv -wdt $3 -hgt $4 -f $6 -o $1_rec.yuv -b $1.bin >>$1.log 