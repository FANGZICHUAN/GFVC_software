#!/bin/sh

./image_codecs/vtm/bin/EncoderAppStatic_VTM22_2_GFVE -c ./image_codecs/vtm/cfg/encoder_intra_vtm.cfg -c $1 --InputFile=$2 --BitstreamFile=$3 --ReconFile=$4>> $5