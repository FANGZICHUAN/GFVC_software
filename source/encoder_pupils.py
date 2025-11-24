import os
import time
import warnings

import cv2
import torch
import imageio
import numpy as np
from copy import copy
from GFVC.utils import *
from GFVC.DAC_utils import *
from GFVC.DAC.metric_utils import eval_metrics
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from argparse import ArgumentParser
from typing import Dict, List
"""add import"""
import mediapipe as mp
from GFVC.Pupil.utils.OneEuroFilter import *
from GFVC.Pupil.utils.EyeFunc import *
from GFVC.Pupil.utils import eye
from GFVC.Pupil.utils.codecs_utils import *
import shutil





if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = ArgumentParser()
    parser.add_argument("--original_seq", default='/home/mmc/datasets/testing_sequence_30/VOXCELEB_013_256x256_25fps_8bit_444.rgb', type=str,
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
    parser.add_argument("--add", default='True', type=str, help="add the gfvc feature(AE)")
    # When the encoding method is AE, the add flag indicates whether to entropy encode the pupil coordinates together with the GFVC features.
    # If add is True, the pupil coordinates and GFVC features are encoded together into a binary file; otherwise, they are not.
    parser.add_argument("--adaptive", default='ON', type=str, help="OFF:non-adaptive;ON:adaptive")
    # Indicates whether to enable the adaptive mechanism,
    # which determines the value of pupilPresentIdx based on the data.

    opt = parser.parse_args()
    frames = int(opt.encoding_frames)
    width = opt.seq_width
    height = opt.seq_width
    quantization_factor = opt.quantization_factor
    q_step = 1 << opt.quantization_factor
    qp = opt.iframe_qp
    original_seq = opt.original_seq
    iframe_format = opt.iframe_format
    device = opt.device
    feature_coding_mode = opt.feature_coding_mode
    gfvc = opt.gfvc

    adaptive = opt.adaptive
    if adaptive == 'ON':
        ada_str = 'adaptive'
    else:
        ada_str = 'non_adaptive'
    add = opt.add == 'True'

    if not torch.cuda.is_available():
        device = 'cpu'

    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]

    model_name = 'Pupil'
    gfvc_dirname = f'../experiment/{gfvc.upper()}/Iframe_{iframe_format}/{width}x{height}/{feature_coding_mode}'
    model_dirname = f'{gfvc_dirname}/{model_name}'
    ###################
    start = time.time()
    if feature_coding_mode == 'AE':
        kp_output_path = model_dirname + '/kp/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/'  # OutPut directory for motion keypoints
        # shutil.rmtree(kp_output_path)
        os.makedirs(kp_output_path, exist_ok=True)

        gfvc_input_path = gfvc_dirname + '/enc/' + seq + '_qp' + str(qp) + '/frame0000_rec'     # the path for first frame

        # enc_output_path = model_dirname + '/enc/' + seq + '_Qstep_' + str(q_step) + '/'  ## OutPut path for encoded reference frame
        # os.makedirs(enc_output_path, exist_ok=True)

    elif feature_coding_mode == 'SEI':
        sei_cfg_output_path = model_dirname + '/cfg/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/'  # OutPut directory for SEI cfg
        os.makedirs(sei_cfg_output_path, exist_ok=True)

        sei_enc_output_path = model_dirname + '/enc/' + seq + '_qp' + str(qp) + '_Qstep' + str(q_step) + '/'  # OutPut directory for SEI bin
        os.makedirs(sei_enc_output_path, exist_ok=True)

        enc_output_path = gfvc_dirname + '/enc/' + seq + '_qp' + str(qp) + '/'

        # gfvc_input_path = gfvc_dirname + '/enc/' + seq + '_qp' + str(qp) + '/allframe_org_rec'     # the path for first frame
    else:
        raise ValueError("The feature_coding_mode can only be AE or AEI.")

    # get the first frame
    if feature_coding_mode == 'AE':
        if iframe_format == 'YUV420':
            listR, listG, listB = read_yuv420(gfvc_input_path + '.yuv', width, height, 1)
        elif iframe_format == 'RGB444':
            listR, listG, listB = raw_reader_planar(gfvc_input_path + '.rgb', width, height, 1)
        else:
            raise ValueError('iframe_format can only be YUV420 or RGB444')

        frame0 = np.array([listR[0], listG[0], listB[0]])
        frame0 = frame0.transpose([1, 2, 0])

    # get original video
    listR, listG, listB = raw_reader_planar(original_seq, width, height, frames)
    f_org = open(original_seq, 'rb')

    if feature_coding_mode == 'AE':
        if add:
            gfvc_kp_path0 = f'../experiment/{gfvc.upper()}/Iframe_{iframe_format}/{width}x{height}/AE/kp/{seq}_qp{qp}/frame'
            kp_coder = PupilsKPEncoder(kp_output_path, q_step, device=device, gfvc_kp_path=gfvc_kp_path0+'0.txt')
        else:
            kp_coder = PupilsKPEncoder(kp_output_path, q_step, device=device)
    elif feature_coding_mode == 'SEI':
        sei_coder = PupilsSEIEncoder(precision_factor=quantization_factor, adaptive=adaptive)
    parse = {# The parameters of the one-euro filter.
        'mincutoff' : 2,
        'beta' : 0.0007,
        'dcutoff' : 3,
        'te' : 1.0 / 25
    }
    lmks_detect = PupilsDetection(**parse)

    """encode"""
    if feature_coding_mode == 'AE':
        with torch.no_grad():
            for frame_idx in tqdm(range(0,frames)):
                # detect lmks
                if frame_idx == 0:
                    lmks_detect.detect(frame0)
                    pupils = np.array(lmks_detect.cvtCoord())
                    # encoder
                    kp_frame = kp_coder.get_kp_list(pupils, frame_idx)
                    kp_coder.rec_sem.append(kp_frame)
                else:
                    current_frame = np.array([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
                    current_frame = np.transpose(current_frame, [1, 2, 0])
                    lmks_detect.detect(current_frame)
                    pupils = np.array(lmks_detect.cvtCoord())

                    # encoder
                    kp_frame = kp_coder.get_kp_list(pupils, frame_idx)
                    if add:
                        gfvc_kp_path = gfvc_kp_path0 + f'{frame_idx}.txt'
                        rec_kp_frame, kp_bits= kp_coder.encode_kp(kp_frame, frame_idx, gfvc_kp_path)
                    else:
                        rec_kp_frame, kp_bits = kp_coder.encode_kp(kp_frame, frame_idx)

    elif feature_coding_mode == 'SEI':
        with torch.no_grad():
            for frame_idx in tqdm(range(0, frames)):
                # detect lmks
                # if frame_idx == 0:
                #     lmks_detect.detect(frame0)
                #     lmks_detect.cvtCoord()
                # else:
                current_frame = np.array([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
                current_frame = np.transpose(current_frame, [1, 2, 0])
                lmks_detect.detect(current_frame)
                lmks_detect.cvtCoord()
            sei_coder.lmks2Cfg(lmks_detect.lmks['pupils'])
            sei_cfg_output = sei_cfg_output_path + 'generative_face_video_enhancement.cfg'
            sei_coder.update_cfg('./GFVC/Pupil/config/base.cfg', sei_cfg_output )

            try:
                os.system("./image_codecs/vtm/encode_GFVE.sh " + sei_cfg_output
                          + ' ' + enc_output_path + 'allframe_org.yuv '
                          + sei_enc_output_path + 'gfveSEI.bin '
                          + sei_enc_output_path + 'rec.yuv '
                          + sei_enc_output_path + 'Enc.txt' )
            except Exception as e:
                print("Command execution failed:", str(e))

    end = time.time()


