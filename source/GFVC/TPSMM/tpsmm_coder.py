import os
import yaml
import torch
import numpy as np

# 导入 TPS-MM 的核心网络
from GFVC.TPSMM.modules.keypoint_detector import KPDetector
from GFVC.TPSMM.modules.inpainting_network import InpaintingNetwork
from GFVC.TPSMM.modules.dense_motion import DenseMotionNetwork

# 导入 GFVC 软件已有的特征量化与算术编解码模块
from arithmetic.value_encoder import final_encoder_expgolomb
from arithmetic.value_decoder import final_decoder_expgolomb, data_convert_inverse_expgolomb

class TPSMMCoder:
    def __init__(self, model_config_path, model_checkpoint_path, kp_output_dir=None, q_step=None, device='cuda'):
        self.device = device
        self.q_step = q_step if q_step is not None else [128] # TPS-MM特征值通常在[-1, 1]之间，128是一个合理的初始缩放倍数
        self.kp_output_dir = kp_output_dir

        # 内部状态变量
        self.reference = None
        self.kp_reference = None
        self.quantized_ref_kp = None # 保存量化后的参考帧关键点，用于计算残差
        self.rec_sem = []

        # 当 model_config_path 为 None 时，跳过网络初始化（仅用于特征解码的轻量实例）
        if model_config_path is None:
            self.kp_extractor = None
            self.dense_motion_network = None
            self.inpainting_network = None
            return

        # 1. 读取 TPS-MM 配置文件
        with open(model_config_path) as f:
            config = yaml.safe_load(f)

        # 2. 初始化 TPS-MM 的三大核心网络模块
        self.kp_extractor = KPDetector(**config['model_params']['common_params']).to(self.device)
        self.dense_motion_network = DenseMotionNetwork(**config['model_params']['dense_motion_params'],
                                                       **config['model_params']['common_params']).to(self.device)
        self.inpainting_network = InpaintingNetwork(**config['model_params']['generator_params'],
                                                    **config['model_params']['common_params']).to(self.device)
        
        # 3. 加载预训练权重
        if model_checkpoint_path is not None:
            checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
            self.kp_extractor.load_state_dict(checkpoint['kp_detector'])
            self.dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
            self.inpainting_network.load_state_dict(checkpoint['inpainting_network'])
            
        self.kp_extractor.eval()
        self.dense_motion_network.eval()
        self.inpainting_network.eval()

    def reference_frame_(self, reference):
        """保存 I 帧作为参考"""
        self.reference = reference

    def kp_reference_(self, kp_reference):
        """保存 I 帧的特征，并计算其量化值"""
        self.kp_reference = kp_reference
        # 将浮点型的参考帧特征放大并取整
        self.quantized_ref_kp = torch.round(self.kp_reference['fg_kp'] * self.q_step[0])

    def analysis_model(self, frame):
        """编码端：提取 TPS 控制点特征"""
        with torch.no_grad():
            kp_value = self.kp_extractor(frame)
        return kp_value

    def get_kp_list(self, kp_dict, frame_idx, save=False):
        """兼容接口：简单返回提取出的字典，并在 I 帧时顺便初始化参考状态"""
        if frame_idx == 0:
            self.kp_reference_(kp_dict) # 拦截 I 帧特征，补全 kp_decoder 的参考状态
        return kp_dict

    def predict_inter_frame(self, kp_inter, frame_idx):
        """解码端：利用解码出的控制点生成画面"""
        with torch.no_grad():
            # TPS-MM 推理管线：源图像 + 源控制点 + 驱动控制点 -> 密集光流 -> Inpainting渲染
            dense_motion = self.dense_motion_network(source_image=self.reference, 
                                                     kp_driving=kp_inter, 
                                                     kp_source=self.kp_reference)
            out = self.inpainting_network(self.reference, dense_motion)
            predicted_frame = out['prediction']
        return predicted_frame

    # ---------------------------------------------------------------------------------
    # 特征量化与算术编码实现
    # ---------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    # 替换为极其稳定、绝对不会卡死、没有 Padding 垃圾的安全编解码方案
    # ---------------------------------------------------------------------------------
    def encode_kp(self, kp_frame, frame_idx):
        """量化、计算残差并编码为二进制文件"""
        # 1. 提取当前帧浮点特征并量化
        current_kp_value = kp_frame['fg_kp']
        quantized_current_kp = torch.round(current_kp_value * self.q_step[0])

        # 2. 计算残差
        residual = quantized_current_kp - self.quantized_ref_kp

        # 3. 转成 list of ints
        residual_list = residual.flatten().cpu().numpy().astype(np.int16).tolist()

        # 4. 使用 Exp-Golomb 编码
        bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
        final_encoder_expgolomb(residual_list, bin_path)

        # 计算文件大小 (比特数)
        kp_bits = os.path.getsize(bin_path) * 8

        return kp_frame, kp_bits

    def decode_kp(self, frame_idx):
        """从二进制文件解码、反量化并恢复特征"""
        if self.quantized_ref_kp is None:
            if len(self.rec_sem) > 0:
                self.kp_reference_(self.rec_sem[0])
            else:
                raise ValueError("参考帧特征未初始化，无法计算残差！")

        # 1. 读取 .bin 文件并使用 Exp-Golomb 解码
        bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
        
        try:
            kp_dec = final_decoder_expgolomb(bin_path)
            residual_list = data_convert_inverse_expgolomb(kp_dec)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到 {bin_path}，请确保你先使用最新的代码重新运行了编码器(encoder)！")
            
        ref_kp_list = self.quantized_ref_kp.flatten().int().cpu().numpy().tolist()
        
        # 2. 重建量化后的特征值
        reconstructed_quantized_list = [ref + res for ref, res in zip(ref_kp_list, residual_list)]
        
        # 3. 恢复严格的张量形状 (1, 50, 2)
        num_tps = len(reconstructed_quantized_list) // 2
        reconstructed_tensor = torch.tensor(reconstructed_quantized_list).view(1, num_tps, 2).float().to(self.device)
        
        # 4. 反量化恢复浮点
        reconstructed_value = reconstructed_tensor / self.q_step[0]
        
        dec_kp = {'fg_kp': reconstructed_value}
        kp_bits = os.path.getsize(bin_path) * 8
        
        return dec_kp, kp_bits
