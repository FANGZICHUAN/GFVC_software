import os
import yaml
import torch
import numpy as np

# 导入 TPS-MM 的核心网络
from GFVC.TPSMM.modules.keypoint_detector import KPDetector
from GFVC.TPSMM.modules.inpainting_network import InpaintingNetwork
from GFVC.TPSMM.modules.dense_motion import DenseMotionNetwork

# 导入 GFVC 软件已有的特征量化与算术编解码模块
from arithmetic.value_encoder import FastEnc
from arithmetic.value_decoder import final_decoder_expgolomb, data_convert_inverse_expgolomb

class TPSMMCoder:
    def __init__(self, model_config_path, model_checkpoint_path, kp_output_dir=None, q_step=None, device='cuda'):
        self.device = device
        self.q_step = q_step if q_step is not None else [128] # TPS-MM特征值通常在[-1, 1]之间，128是一个合理的初始缩放倍数
        self.kp_output_dir = kp_output_dir
        
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

        # 内部状态变量
        self.reference = None
        self.kp_reference = None
        self.quantized_ref_kp = None # 保存量化后的参考帧关键点，用于计算残差
        self.rec_sem = []

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
        """兼容接口：简单返回提取出的字典"""
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
    def encode_kp(self, kp_frame, frame_idx):
        """量化、计算残差并编码为二进制文件"""
        # 1. 提取当前帧浮点特征并量化
        current_kp_value = kp_frame['fg_kp']
        quantized_current_kp = torch.round(current_kp_value * self.q_step[0])
        
        # 2. 计算残差 (当前帧特征 - 参考帧特征)
        residual = quantized_current_kp - self.quantized_ref_kp
        
        # 3. 展平为一维整数列表，准备送入你代码里的算术编码器
        residual_list = residual.flatten().int().cpu().numpy().tolist()
        
        # 4. 调用 GFVC 的 0阶指数哥伦布编码器
        bin_path = os.path.join(self.kp_output_dir, str(frame_idx).zfill(4) + '.bin')
        fast_enc = FastEnc()
        fast_enc.exp_golomb_coder(residual_list, bin_path)
        
        # 计算文件大小 (比特数)
        kp_bits = os.path.getsize(bin_path) * 8
        
        return kp_frame, kp_bits

    def decode_kp(self, frame_idx):
        """从二进制文件解码、反量化并恢复特征"""
        bin_path = os.path.join(self.kp_output_dir, str(frame_idx).zfill(4) + '.bin')
        
        # 1. 读取二进制文件并解码出 0/1 序列
        datares_rec = final_decoder_expgolomb(bin_path)
        
        # 2. 将 0/1 序列逆向转换为一维残差数值列表
        residual_list = data_convert_inverse_expgolomb(datares_rec)
        
        # 3. 加上参考帧的量化特征，重建当前帧的量化特征
        ref_kp_list = self.quantized_ref_kp.flatten().int().cpu().numpy().tolist()
        reconstructed_quantized_list = np.array(ref_kp_list) + np.array(residual_list)
        
        # 4. 恢复张量形状 (1, num_tps, 2)
        num_tps = len(reconstructed_quantized_list) // 2
        reconstructed_tensor = torch.tensor(reconstructed_quantized_list).view(1, num_tps, 2).float().to(self.device)
        
        # 5. 反量化 (除以 q_step) 恢复为浮点数
        reconstructed_value = reconstructed_tensor / self.q_step[0]
        
        # 封装回 TPS-MM 需要的字典格式
        dec_kp = {'fg_kp': reconstructed_value}
        
        kp_bits = os.path.getsize(bin_path) * 8
        
        return dec_kp, kp_bits