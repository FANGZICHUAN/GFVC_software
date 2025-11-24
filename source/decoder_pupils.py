import os
import time
import json
import torch
import imageio
import numpy as np
from GFVC.utils import *
from GFVC.codecs_utils import *
from argparse import ArgumentParser
from skimage.transform import resize
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from typing import Dict, List
from copy import copy

"""add import"""
from GFVC.Pupil.modules.generator import OcclusionAwareGenerator
from GFVC.Pupil.logger import Logger
import mediapipe as mp
import torchvision.transforms as transforms

from GFVC.Pupil.utils.OneEuroFilter import OneEuroFilter
import cv2 as cv

from GFVC.Pupil.utils.EyeFunc import *
from GFVC.Pupil.utils.codecs_utils import *
import warnings


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='/home/mmc/datasets/testing_sequence_30/VOXCELEB_013_256x256_25_8bit_444.rgb', type=str,
                        help="path to the input testing sequence")
    parser.add_argument("--gfvc", default='fomm', help="the mode of gfvc algorithm")
    parser.add_argument("--encoding_frames", default=250, type=int, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, type=int, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, type=int, help="the height of encoding frames")
    parser.add_argument("--quantization_factor", default=5, type=int,
                        help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=22, type=int, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,
                        help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--device", default='cuda', type=str, help="execution device: [cpu, cuda]")
    parser.add_argument("--feature_coding_mode", default='SEI', type=str,
                        help="the implementation algorithm for supplement enhancement informantion on the basis of GFVC software")
    parser.add_argument("--adaptive", default='ON', type=str, help="switch the model,OFF:non-adaptive;ON:adaptive")


    opt = parser.parse_args()
    frames = int(opt.encoding_frames)
    width = opt.seq_width
    height = opt.seq_width
    q_step = 1 << opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    device = opt.device
    feature_coding_mode = opt.feature_coding_mode
    gfvc = opt.gfvc

    if opt.adaptive == 'ON':
        ada_str = 'adaptive'
    else:
        ada_str = 'non_adaptive'

    if not torch.cuda.is_available():
        device = 'cpu'

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    ## FOM
    model_name = "Pupil"

    model_config_path = f'./GFVC/{model_name}/checkpoint/{model_name}-256.yaml'
    model_left_checkpoint_path = f'./GFVC/{model_name}/checkpoint/Pupil-left-checkpoint.pth.tar'
    model_right_checkpoint_path = f'./GFVC/{model_name}/checkpoint/Pupil-right-checkpoint.pth.tar'
    model_dirname = f'../experiment/{gfvc.upper()}/Iframe_{iframe_format}/{width}x{height}/{feature_coding_mode}/{model_name}'
    GFVC_dec_seq = f'../experiment/{gfvc.upper()}/Iframe_{iframe_format}/{width}x{height}/{feature_coding_mode}/dec/{seq}_qp{qp}.rgb'
    ###################################################
    if feature_coding_mode == 'AE':
        kp_input_path = model_dirname + '/kp/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/'  # InPut directory for motion keypoints

        dec_output_path = model_dirname + '/dec/'
        os.makedirs(dec_output_path, exist_ok=True)  # the real decoded video
        dec_sequence_path_rgb = dec_output_path + seq + f'_qp{qp}' + '_Qstep' + str(q_step) + '.rgb'
        dec_sequence_path_mp4 = dec_output_path + seq + f'_qp{qp}' + '_Qstep' + str(q_step) + '.mp4'

        dir_bit = model_dirname + '/resultBit/'
        os.makedirs(dir_bit, exist_ok=True)
        dir_bit = dir_bit  + seq + '_Qstep' + str(q_step)

        # enc_input_path = model_dirname + '/enc/' + seq + '_Qstep_' + str(q_step) + '/'  ## InPut path for encoded reference frame
    elif feature_coding_mode == 'SEI':
        sei_cfg_output_path = model_dirname + '/outSEI/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/'
        os.makedirs(sei_cfg_output_path, exist_ok=True)
        sei_bin_intput_path = model_dirname + '/enc/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/gfveSEI.bin'

        dec_output_path = model_dirname + '/dec/' + seq + f'_qp{qp}' + '_Qstep' + str(q_step) + f'_{ada_str}/'
        os.makedirs(dec_output_path, exist_ok=True)  # the real decoded video
        dec_sequence_path_rgb = dec_output_path + seq + f'_qp{qp}' + '_Qstep' + str(q_step) + f'_{ada_str}.rgb'
        dec_sequence_path_mp4 = dec_output_path + seq + f'_qp{qp}' + '_Qstep' + str(q_step) + f'_{ada_str}.mp4'

        dir_bit = model_dirname + '/resultBit/'
        os.makedirs(dir_bit, exist_ok=True)
        dir_bit = dir_bit + seq + '_Qstep' + str(q_step) + f'_{ada_str}'
    else:
        raise ValueError("The feature_coding_mode can only be AE or AEI.")

    # ref_input_path = model_dirname + '/enc/' + seq + '/'
    # os.makedirs(ref_input_path, exist_ok=True)  # the frames to be compressed by vtm



    # Initialize the output file

    f_dec = open(dec_sequence_path_rgb, 'wb')

    start = time.time()
    sum_bits = 0

    # Keypoint Decoder
    if feature_coding_mode == 'AE':
        kp_decoder = PupilsKPDecoder(kp_input_path, q_step)
    elif feature_coding_mode == 'SEI':
        kp_decoder = PupilsSEIDecoder()
    else:
        raise ValueError("The feature_coding_mode can only be AE or AEI.")
    # Reconstruction models
    inter_model = EyeGenerate(model_config_path=model_config_path,
                              model_left_checkpoint_path=model_left_checkpoint_path,
                              model_right_checkpoint_path=model_right_checkpoint_path,
                              device=device)

    out_video = []  # Output an mp4 video for sanity check

    listR, listG, listB = raw_reader_planar(GFVC_dec_seq, width, height, frames)

    if feature_coding_mode == 'AE':
        with torch.no_grad():
            for frame_idx in tqdm(range(0, frames)):
                current_frame = np.array([listR[frame_idx], listG[frame_idx], listB[frame_idx]]) # 3*H*W
                current_frame = np.transpose(current_frame, [1, 2, 0])                      # H*W*3, RGB
                if frame_idx == 0:
                    # 检测landmarks
                    inter_model.lmks_detect.detect(current_frame)
                    pupils = np.array(inter_model.lmks_detect.cvtCoord())
                    kp_frame = kp_decoder.get_kp_list(pupils, frame_idx)
                    # Encode to binary string
                    kp_decoder.app_rec_sem(kp_frame)
                else:
                    pupil = {}
                    pupil['left'], pupil['right'], kp_bits = kp_decoder.decode_kp(frame_idx)
                    sum_bits += kp_bits
                    current_frame = inter_model.process_frame(pupil, current_frame)
                # Save to output file
                out_video.append(current_frame)
                current_frame.tofile(f_dec)

    elif feature_coding_mode == 'SEI':
        try:
            os.system("./image_codecs/vtm/decode_GFVE.sh " + sei_bin_intput_path + ' ' + dec_output_path + 'dec.yuv '
                      + sei_cfg_output_path + 'outSEI.txt')
        except Exception as e:
            print("Command execution failed:", str(e))
        #调用示例
        split_text_file_gfve(sei_cfg_output_path + 'outSEI.txt', sei_cfg_output_path)
        with torch.no_grad():
            for frame_idx in tqdm(range(0, frames)):
                current_frame = np.array([listR[frame_idx], listG[frame_idx], listB[frame_idx]])  # 3*H*W
                current_frame = np.transpose(current_frame, [1, 2, 0])  # H*W*3, RGB
                pupil = kp_decoder.dec_sei(sei_cfg_output_path + f'gfve_{frame_idx}.txt')

                current_frame = inter_model.process_frame(pupil, current_frame)
                out_video.append(current_frame)
                current_frame.tofile(f_dec)

        sum_bits += os.path.getsize(sei_bin_intput_path) * 8

    else:
        raise ValueError("The feature_coding_mode can only be AE or AEI.")
    f_dec.close()
    end = time.time()
    imageio.mimsave(dec_sequence_path_mp4, out_video, fps=25.0)
    totalResult = np.zeros((1, 2))
    totalResult[0][0] = sum_bits
    totalResult[0][1] = end - start

    np.savetxt(dir_bit, totalResult, fmt='%.5f')
    # print("rec={}\npupils={}".format(kp_decoder.rec, result_pupils))


