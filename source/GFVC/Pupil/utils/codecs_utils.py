import os
import time
import warnings
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

from GFVC.Pupil.modules.generator import OcclusionAwareGenerator
from GFVC.Pupil.logger import Logger
import torchvision.transforms as transforms
import cv2 as cv
import configparser

def read_yuv420(file_path, width, height, num_frames):
    frame_size = width * height * 3 // 2  # YUV420的单帧大小
    r_list = []  # 用于存储所有帧的R通道
    g_list = []  # 用于存储所有帧的G通道
    b_list = []  # 用于存储所有帧的B通道

    # 打开YUV文件
    with open(file_path, 'rb') as f:
        for i in range(num_frames):
            # oneframe_I420 = np.zeros((height * 3 // 2, width), np.uint8)
            # for j in range(height * 3 // 2):
            #     for k in range(width):
            #         oneframe_I420[j, k] = int.from_bytes(f.read(1), byteorder='little', signed=False)
            # # 将YUV转换为RGB
            # oneframe_RGB = cv.cvtColor(oneframe_I420, cv.COLOR_YUV2RGB_I420)

            # 读取单帧YUV数据
            yuv_frame = np.frombuffer(f.read(frame_size), dtype=np.uint8)

            if len(yuv_frame) < frame_size:
                print("读取到文件末尾，停止")
                break

            # 将YUV数据转换为Y, U, V平面
            y = yuv_frame[0:width * height].reshape((height, width))
            u = yuv_frame[width * height:width * height + (width // 2) * (height // 2)].reshape(
                (height // 4, width))
            v = yuv_frame[width * height + (width // 2) * (height // 2):].reshape((height // 4, width))

            yuv420p = cv.vconcat([y,u,v])

            # 将YUV转换为RGB
            oneframe_RGB = cv.cvtColor(yuv420p, cv.COLOR_YUV2RGB_I420)

            # 提取R, G, B通道
            r_channel, g_channel, b_channel = cv.split(oneframe_RGB)

            # 将R, G, B通道添加到各自的列表中
            r_list.append(r_channel)
            g_list.append(g_channel)
            b_list.append(b_channel)

    return r_list, g_list, b_list


def signed_to_unsigned(n):
    """
    将有符号整数转换为无符号整数，以便进行指数哥伦布编码。

    参数：
    - n: 有符号整数

    返回：
    - 无符号整数
    """
    if n >= 0:
        return 2 * n
    else:
        return -2 * n - 1


def unsigned_to_signed(n):
    """
    将无符号整数转换回有符号整数。

    参数：
    - n: 无符号整数

    返回：
    - 有符号整数
    """
    if n % 2 == 0:
        return n // 2
    else:
        return -(n // 2) - 1


def exp_golomb_encode_signed(n):
    """
    对有符号整数进行指数哥伦布编码。

    参数：
    - n: 有符号整数

    返回：
    - 编码后的二进制字符串
    """
    unsigned_n = signed_to_unsigned(n)

    # 使用无符号整数的指数哥伦布编码
    return exp_golomb_encode(unsigned_n)


def exp_golomb_decode_signed(encoded):
    """
    对指数哥伦布编码的二进制字符串进行解码，返回有符号整数。

    参数：
    - encoded: 编码后的二进制字符串

    返回：
    - 解码后的有符号整数
    """
    unsigned_n, length = exp_golomb_decode(encoded)

    return unsigned_to_signed(unsigned_n), length


def exp_golomb_encode(n):
    """
    指数哥伦布编码（无符号整数）。

    参数：
    - n: 无符号整数

    返回：
    - 编码后的二进制字符串
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    # Find the exponent of the highest power of 2 <= n + 1
    m = 0
    while (1 << m) <= n + 1:
        m += 1
    m -= 1

    # The prefix is m zeroes
    prefix = '0' * m

    # The suffix is the binary representation of n+1, using (m+1) bits
    suffix = format(n + 1, '0{}b'.format(m + 1))

    # Concatenate prefix and suffix
    return prefix + suffix


def exp_golomb_decode(encoded):
    """
    指数哥伦布解码（无符号整数）。

    参数：
    - encoded: 编码后的二进制字符串

    返回：
    - 解码后的无符号整数
    - 该码字长度
    """
    # Count the number of leading zeroes
    m = 0
    while m < len(encoded) and encoded[m] == '0':
        m += 1

    # Extract the suffix (m+1 bits after the leading zeroes)
    suffix = encoded[m:m + m + 1]
    length = len(suffix) * 2 -1

    # Convert the suffix to an integer and subtract 1 to get the original number
    n = int(suffix, 2) - 1

    return n, length

def write_binary_string_to_bin_file(binary_string, filename):
    """
    将二进制字符串写入二进制文件。

    参数：
    - binary_string: 例如 '01'，这是一个表示二进制数据的字符串
    - filename: 二进制文件名（例如 'output.bin'）
    """
    # 将二进制字符串转换为整数
    binary_data = int(binary_string, 2)

    # 计算需要多少字节来存储这个整数
    byte_length = (binary_data.bit_length() + 7) // 8

    # 将整数转换为字节数据
    byte_data = binary_data.to_bytes(byte_length, byteorder='big')

    # 打开二进制文件以写入 ('wb' 模式)
    with open(filename, 'wb') as bin_file:
        bin_file.write(byte_data)
"""*****************************************************************************************************************************************************************"""
class PupilsKPEncoder:
    def __init__(self, kp_output_dir: str, q_step: int = 64, num_kp=2, device='cpu', gfvc='fv2v', gfvc_kp_path=None):
        self.gfvc = gfvc
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.device = device
        self.num_kp = num_kp
        self.rec_sem = []
        self.bits = []
        self.kp_diff = []
        self.kp_list = []

        # get gfvc kp
        if gfvc_kp_path == None:
            self.switch = 0
        else:
            self.switch = 1
            self.gfvc_rec_sem = []
            with open(gfvc_kp_path, 'r') as f:
                kp_gfvc = f.readlines()
            kp_gfvc = eval('[%s]' % repr(kp_gfvc).replace('[', '').replace(']', '').replace('\'', '').replace('\\n', ''))

            self.gfvc_rec_sem.append(kp_gfvc)


    def get_kp_list(self, kp_value, frame_idx: int, save=True) -> List[str]:
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        if save:
            with open(self.kp_output_dir + '/frame' + str(frame_idx) + '.txt', 'w') as f:
                f.write(kp_value_list)

        kp_value_frame = json.loads(kp_value_list)  ###20
        kp_value_frame = eval('[%s]' % repr(kp_value_frame).replace('[', '').replace(']', ''))

        # 记录所有的kp，后面记录在txt里面
        self.kp_list += kp_value_frame
        return kp_value_frame

    def encode_kp(self, kp_list: List[str], frame_idx: int, gfvc_kp_path=None):
        ## residual
        kp_difference = (np.array(kp_list) - np.array(self.rec_sem[-1])).tolist()
        ## quantization

        kp_difference = [int(i) for i in np.round(np.array(kp_difference) * self.q_step)]

        if self.switch != 0:
        # get gfvc kp
            with open(gfvc_kp_path, 'r') as f:
                kp_gfvc = f.readlines()
            kp_gfvc = eval('[%s]' % repr(kp_gfvc).replace('[', '').replace(']', '').replace('\'', '').replace('\\n', ''))
            kp_difference_gfvc = (np.array(kp_gfvc) - np.array(self.gfvc_rec_sem[-1])).tolist()
            kp_difference_gfvc = [int(i) for i in np.round(np.array(kp_difference_gfvc) * (1<<8))]
            kp_difference = kp_difference_gfvc + kp_difference

        #
        # if self.switch == 0:
        #     self.kp_diff += kp_difference

        # 编码并保存为bin文件
        bin_file = self.kp_output_dir + 'frame' + str(frame_idx).zfill(4) + '.bin'
        final_encoder_expgolomb(kp_difference, bin_file)
        # 计算bit数
        bits = os.path.getsize(bin_file) * 8
        # 统计每帧的bit数
        self.bits.append(bits)
        # 解码
        #### decoding for residual
        kp_dec = final_decoder_expgolomb(bin_file)
        kp_difference_dec = data_convert_inverse_expgolomb(kp_dec)

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
        if self.switch != 0:
            kp_difference_dec_gfvc = kp_difference_dec[:-4]
            kp_difference_dec = kp_difference_dec[-4:]
            kp_difference_dec_gfvc = np.array(kp_difference_dec_gfvc) / (1<<8)
            kp_integer = (np.array(self.gfvc_rec_sem[-1])+np.array(kp_difference_dec_gfvc) ).tolist()  #######
            self.gfvc_rec_sem.append(kp_integer)

        kp_difference_dec = np.array(kp_difference_dec) / self.q_step

        kp_integer, kp_value = listformat_kp_Pupils(self.rec_sem[-1], kp_difference_dec)  #######

        self.rec_sem.append(kp_integer)

        if self.switch == 0:
            kp_inter_frame = {}
            kp_value = json.loads(kp_value)
            kp_current_value = torch.Tensor(kp_value).reshape((1, -1, 2)).to(self.device)
            kp_inter_frame['value'] = kp_current_value
            # reconstruct the KPs
            return kp_inter_frame, bits
        return 0, bits

    def encode_all_kp(self):
        bin_file = self.kp_output_dir + 'frame_all.bin'

        # kp_temp = np.zeros_like(self.kp_diff)
        # kp_temp = kp_temp.tolist()
        # final_encoder_expgolomb(kp_temp, bin_file)
        final_encoder_expgolomb(self.kp_diff, bin_file)

        bits = os.path.getsize(bin_file) * 8

        kp_list = str(self.kp_list)
        kp_list = "".join(kp_list.split())
        with open(self.kp_output_dir + '/frame_all.txt', 'w') as f:
            f.write(kp_list)

        return bits



class PupilsKPDecoder:
    def __init__(self, kp_output_dir: str, q_step: int = 64, num_kp=2, switch=1):
        self.switch = switch
        self.kp_output_dir = kp_output_dir
        self.q_step = q_step
        self.num_kp = num_kp
        self.rec_sem = []

        # 一欧元滤波
        mincutoff = 0.5  # 最小截止频率
        beta = 0.5  # 速度因子
        dcutoff = 1  # 截止频率
        self.te = 1.0 / 25
        self.filter = {}

        self.filter['left'] = OneEuroFilter(self.te, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self.filter['right'] = OneEuroFilter(self.te, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)


    def app_rec_sem(self, rec_sem):
        self.rec_sem.append(rec_sem)
        self.filter['left'].predict(np.array(rec_sem[-4:-2]), self.te)
        self.filter['right'].predict(np.array(rec_sem[-2:]), self.te)

    def get_kp_list(self, kp_value, frame_idx: int) -> List[str]:
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        with open(self.kp_output_dir + 'frame' + str(frame_idx) + '.txt', 'w') as f:
            f.write(kp_value_list)

        kp_value_frame = json.loads(kp_value_list)  ###20
        kp_value_frame = eval('[%s]' % repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame

    def decode_kp(self, frame_idx: int) -> None:
        frame_index = str(frame_idx).zfill(4)
        bin_file = self.kp_output_dir + 'frame' + frame_index + '.bin'
        bits = os.path.getsize(bin_file) * 8
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        if self.switch != 0:
            kp_difference = kp_difference[-4:]
        ## inverse quantization
        kp_difference_dec = [i / self.q_step for i in kp_difference]
        kp_difference_dec = eval('[%s]' % repr(kp_difference_dec).replace('[', '').replace(']', ''))

        kp_previous = eval('[%s]' % repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))

        kp_integer, kp_value = listformat_kp_Pupils(kp_previous, kp_difference_dec[-4:])  #######
        self.rec_sem.append(kp_integer)

        kp_left = np.array([kp_integer[0], kp_integer[1]])
        kp_right = np.array([kp_integer[2], kp_integer[3]])

        self.filter['left'].predict(kp_left, self.te)
        self.filter['right'].predict(kp_right, self.te)

        # kp_inter_frame = {}
        # kp_value = json.loads(kp_value)
        # kp_current_value = torch.Tensor(kp_value).reshape((1, self.num_kp, 2)).to(device)
        # kp_inter_frame['value'] = kp_current_value
        return kp_left, kp_right, bits

    def decode_kp_all(self, num_frames):
        bin_file = self.kp_output_dir + 'frame_all.bin'
        bits = os.path.getsize(bin_file) * 8
        kp_dec = final_decoder_expgolomb(bin_file)
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        kp_difference_dec_all = [i / self.q_step for i in kp_difference]

        ## decoding residual
        for idx in range(num_frames-1):
            if len(kp_difference_dec_all[idx*4:]) < 4:
                raise ValueError("the num_frames is too large")

            kp_difference_dec = kp_difference_dec_all[idx*4:(idx+1)*4]
            kp_difference_dec = eval('[%s]' % repr(kp_difference_dec).replace('[', '').replace(']', ''))

            kp_previous = eval('[%s]' % repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))

            kp_integer, kp_value = listformat_kp_Pupils(kp_previous, kp_difference_dec)  #######
            self.rec_sem.append(kp_integer)

        return bits

    def decode_kp_load(self, frame_idx):
        kp_integer = self.rec_sem[frame_idx]
        kp_left = np.array([kp_integer[0], kp_integer[1]])
        kp_right = np.array([kp_integer[2], kp_integer[3]])

        return kp_left, kp_right

    def load_metadata(self) -> None:
        bin_file = self.kp_output_dir + 'metadata.bin'
        dec_metadata = final_decoder_expgolomb(bin_file)
        metadata = data_convert_inverse_expgolomb(dec_metadata)
        self.ref_frame_idx = [int(i) for i in metadata]
        return os.path.getsize(bin_file) * 8

"""*****************************************************************************************************************************************************************"""
class Pupil:
    '''DAC Decoder models'''

    def __init__(self, dac_config_path: str, dac_checkpoint_path: str, device='cpu') -> None:
        self.device = device
        self.reference_frame = None
        self.kp_reference = None
        with open(dac_config_path) as f:
            config = yaml.safe_load(f)

        self.generator= OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                                  **config['model_params']['common_params'])
        self.generator = self.generator.to(device)
        Logger.load_cpk(dac_checkpoint_path, generator=self.generator)
        self.generator.eval()

"""*****************************************************************************************************************************************************************"""
class PupilsDetection:
    def __init__(self, mincutoff = 1.5, beta = 0.05, dcutoff = 2.5, te = 1.0 / 25):
        # 裁减中心点是否滤波
        self.cropFilter_flag = True
        # 是否固定裁减大小
        # self.size = (32, 32)
        self.size = None
        # 是否滤波
        self.pupilFilter_flag = True
        # 是否归一化到[-1, 1]
        self.norm_flag = True
        # 我们提案的坐标系
        self.ourCoordinate_flag = True


        """初始化"""
        # 配置mediapipe
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

        self.CropEyeImage = CropEyeImage()
        # 数据初始化
        self.lmks = {}
        self.lmks['left'] = []
        self.lmks['right'] = []
        self.lmks['pupils'] = []

        # 一欧元滤波
        self.mincutoff = mincutoff  # 最小截止频率
        self.beta = beta  # 速度因子
        self.dcutoff = dcutoff  # 截止频率
        self.te = te
        self.filter = {}

        self.filter['left'] = OneEuroFilter(self.te, mincutoff=self.mincutoff, beta=self.beta, dcutoff=self.dcutoff)
        self.filter['right'] = OneEuroFilter(self.te, mincutoff=self.mincutoff, beta=self.beta, dcutoff=self.dcutoff)


    def detect(self, current_frame):
        # 检测landmarks
        results = self.face_mesh.process(current_frame)
        landmark_points = results.multi_face_landmarks
        self.h, self.w, _ = current_frame.shape

        self.landmarks_dict = mp_getEyeLandmarks(landmark_points[0], self.w, self.h, flip=1)
        self.crop_dict = self.CropEyeImage.cropEyeImage(current_frame, self.landmarks_dict, w=self.w, h=self.h, size=self.size)

        return self.landmarks_dict, self.crop_dict

    def cvtCoord(self):
        for eye_flag_ in ['left', 'right']:
            img_h, img_w, _ = (self.crop_dict[eye_flag_ + "_eye"]['img']).shape                 # crop img
            eye7 = np.array(self.crop_dict[eye_flag_ + "_eye"]["eye7_norm"], dtype='float32')   # norm lmks-7

            if self.pupilFilter_flag:                                                           # OneEuroFilter
                eye7 = self.filter[eye_flag_].predict(eye7, self.te)

            mul = np.array([[self.w * 1.0 / img_w, 0], [0, self.h * 1.0 / img_h]])
            eye7 = np.matmul(eye7, mul)                                                         # normalize to crop size

            if self.norm_flag:                                                                  # normalize to [-1, 1]
                eye7 = eye7 * 2 - 1


            self.lmks[eye_flag_].append(eye7)
        self.lmks['pupils'].append([self.lmks['left'][-1][6], self.lmks['right'][-1][6]])
        return self.lmks['pupils'][-1]



"""*****************************************************************************************************************************************************************"""
class PupilsSEIEncoder:
    def __init__(self, precision_factor, adaptive='OFF'):
        self.adaptive = adaptive
        self.pre_factor = precision_factor
        self.gfve_pupilPredentIdx = []
        self.gfve_pupilPrecisionFactor = []
        self.gfve_pupilCoordinate = {}
        self.gfve_pupilCoordinate['leftX'] = []
        self.gfve_pupilCoordinate['leftY'] = []
        self.gfve_pupilCoordinate['rightX'] = []
        self.gfve_pupilCoordinate['rightY'] = []


    def lmks2Cfg(self, pupils):
        q_step = 1 << self.pre_factor
        # pupils = [[[x1,y1], [x2,y2]],...]
        for idx, pupil in enumerate(pupils):
            if idx == 0:
                self.gfve_pupilPredentIdx += [int('11', 2)]
                self.gfve_pupilPrecisionFactor.append(self.pre_factor)
                self.base_coordinate = np.array(pupil) # np.array
                for j, e in enumerate(['left', 'right']):
                    for i, c in enumerate(['X', 'Y']):
                        v = pupil[j][i]
                        self.gfve_pupilCoordinate[f'{e}{c}'].append(v)
            else:
                if self.adaptive == 'OFF':
                    self.gfve_pupilPredentIdx += [int('11', 2)]
                elif self.adaptive == 'ON':
                    value = ((np.array(pupil)*q_step + 0.5).round() - (self.base_coordinate*q_step + 0.5).round()).astype(np.int0)
                    ret = value == (np.array(pupil)*0).astype(np.int0)
                    gfve_idx = '0' if ret[0][1] and ret[0][0] else '1'
                    gfve_idx += '0' if ret[1][1] and ret[1][0] else '1'
                    self.gfve_pupilPredentIdx += [int(gfve_idx, 2)]
                    self.base_coordinate = np.array(pupil)

                self.gfve_pupilPrecisionFactor.append(self.pre_factor)
                for j, e in enumerate(['left', 'right']):
                    for i, c in enumerate(['X', 'Y']):
                        v = pupil[j][i]
                        self.gfve_pupilCoordinate[f'{e}{c}'].append(v)

    def prepare_to_code(self):
        self.cfgcontent_coordinate_info = ['SEIGenerativeFaceVideoEnhancementPupilPresentIdx: ' + str(self.list_to_cmd(self.gfve_pupilPredentIdx)) + "\n",
                                           'SEIGenerativeFaceVideoEnhancementPupilCoordinatePrecisionFactor: ' + str(self.list_to_cmd(self.gfve_pupilPrecisionFactor)) + "\n",
                                           'SEIGenerativeFaceVideoEnhancementPupilLeftEyeCoordinateX: ' + str(self.list_to_cmd(self.gfve_pupilCoordinate['leftX'])) + "\n",
                                           'SEIGenerativeFaceVideoEnhancementPupilLeftEyeCoordinateY: ' + str(self.list_to_cmd(self.gfve_pupilCoordinate['leftY'])) + "\n",
                                           'SEIGenerativeFaceVideoEnhancementPupilRightEyeCoordinateX: ' + str(self.list_to_cmd(self.gfve_pupilCoordinate['rightX'])) + "\n",
                                           'SEIGenerativeFaceVideoEnhancementPupilRightEyeCoordinateY: ' + str(self.list_to_cmd(self.gfve_pupilCoordinate['rightY'])) + "\n"]

    def list_to_cmd(self, input):
        return eval('%s' % repr(str(input)).replace('[', '').replace(']', '')
                    .replace('(', '').replace(')', '')
                    .replace(' ','').replace(',', ' '))

    def update_cfg(self, input_path, output_path):
        self.prepare_to_code()
        # 读取原始文件内容
        with open(input_path, 'r') as file:
            lines = file.readlines()
        lines = lines + self.cfgcontent_coordinate_info

        # 写入更新后的内容到输出文件
        with open(output_path, 'w') as file:
            file.writelines(lines)


"""*****************************************************************************************************************************************************************"""
class PupilsSEIDecoder:
    def __init__(self):
        # 一欧元滤波
        mincutoff = 0.5  # 最小截止频率
        beta = 0.5  # 速度因子
        dcutoff = 1  # 截止频率
        self.te = 1.0 / 25
        self.filter = OneEuroFilter(self.te, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)

    def dec_sei(self, input_path):
        current_driving_string = []
        with open(input_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if 'gfve_PupilCoordinate:' in line:
                pupil_coordinate_vector = line.strip('gfve_PupilCoordinate: ')
        coodinate = eval('[%s]' % pupil_coordinate_vector.replace(' ', ','))
        coodinate = self.filter.predict(np.array(coodinate), self.te)
        pupil = {}
        pupil['left'] = coodinate[0:2]
        pupil['right'] = coodinate[2:4]
        return pupil


"""*****************************************************************************************************************************************************************"""
class EyeGenerate:
    def __init__(self, model_config_path=None, model_left_checkpoint_path=None, model_right_checkpoint_path=None, device='cpu'):
        self.gen_left = Pupil(model_config_path, model_left_checkpoint_path, device=device)
        self.gen_right = Pupil(model_config_path, model_right_checkpoint_path, device=device)
        self.device = device
        self.my_transform1 = transforms.ToTensor()

        parse = {
            'mincutoff': 1.5,
            'beta': 0.5,
            'dcutoff': 2,
            'te': 1.0 / 25
        }
        self.lmks_detect = PupilsDetection(**parse)

    def process_frame(self, pupil=None, current_frame=None):
        # 识别
        ref_h, ref_w, _ = current_frame.shape
        self.lmks_detect.detect(current_frame)
        self.lmks_detect.cvtCoord()

        """获取驱动图片的蒙版"""
        mask_dict, maskName = getEyeMask(self.lmks_detect.crop_dict, "BGR")

        for eye in ["right", "left"]:
            kp_source = {}
            kp_driving = {}
            # 注：网络在训的时候使用opencv读取图片，因此网络输入BGR，输出是BGR
            source = cv.cvtColor(self.lmks_detect.crop_dict[eye + "_eye"]['img'], cv.COLOR_RGB2BGR)
            source = copy.deepcopy(source).astype(np.float32)
            img_h, img_w, _ = source.shape

            source = cv.resize(source, dsize=(128, 128))
            source = self.my_transform1(source / 255)
            source = source.unsqueeze(0)

            source = source.to(self.device)

            kp_source['value'] = np.array(self.lmks_detect.lmks[eye][-1])
            kp_driving['value'] = copy.deepcopy(kp_source['value'])
            kp_driving['value'][-1] = pupil[eye]
            kp_source['value'] = self.my_transform1(np.array(kp_source['value']).astype(np.float32)).to(self.device)
            kp_driving['value'] = self.my_transform1(np.array(kp_driving['value']).astype(np.float32)).to(self.device)

            start_xy = self.lmks_detect.crop_dict[eye + "_eye"]['start_xy']

            if eye == "right":
                generator = self.gen_right
            elif eye == "left":
                generator = self.gen_left

            # 生成

            out = generator.generator(source, kp_source=kp_source, kp_driving=kp_driving)
            rec = out['prediction'].data.cpu().numpy()
            rec_img = np.transpose(rec, [0, 2, 3, 1])[0]
            rec_img = np.ascontiguousarray(rec_img * 255, dtype=np.uint8)
            rec_img = cv.resize(rec_img, dsize=(img_w, img_h))
            rec_img = cv.cvtColor(rec_img, cv.COLOR_BGR2RGB)

            """mask"""
            rec_area = copy.deepcopy(self.lmks_detect.crop_dict[eye + '_eye']['img'])
            _mask = copy.deepcopy(mask_dict[eye + '_eye_crop']['mask'])

            rec_eye = cv.bitwise_and(rec_img, rec_img, mask=_mask)
            _mask = cv.bitwise_not(_mask)
            rec_area = cv.bitwise_and(rec_area, rec_area, mask=_mask)
            rec_area = cv.add(rec_area, rec_eye)

            current_frame[start_xy[1]:start_xy[1] + img_h, start_xy[0]:start_xy[0] + img_w] = rec_area
        return current_frame
