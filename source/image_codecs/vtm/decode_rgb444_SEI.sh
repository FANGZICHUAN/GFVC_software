#!/bin/sh

./image_codecs/vtm/bin/DecoderAppStatic_VTM22_2_SEI -b $1.bin -o $1_dec.rgb --OutputColourSpaceConvert=GBRtoRGB --OutputDecodedSEIMessagesFilename=$2 > $1_dec.log
