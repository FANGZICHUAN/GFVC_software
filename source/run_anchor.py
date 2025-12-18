import os
import sys
import numpy as np
import argparse
import time
from tqdm import tqdm

# 复用项目现有的工具函数读取配置
from GFVC.utils import read_config_file

def get_file_size_in_bits(file_path):
    if os.path.exists(file_path):
        return os.path.getsize(file_path) * 8
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../config/cfte.yaml', type=str, help="Path to codec configuration file")
    opt = parser.parse_args()
    
    # 1. 读取配置
    args = read_config_file(opt.config)
    seqlist = args['seq_list']
    
    # [修改点 1] 获取 YUV 序列目录，不再使用 sequence_dir (RGB)
    yuv_sequence_dir = args['yuv_sequence_dir'] 
    
    frames = args['num_frames']
    width = args['width']
    height = args['height']
    qplist = args['qp_list']
    dataset_name = args['dataset']
    
    # 强制将模型名称设为 VVC_Anchor
    model_name = "VVC_Anchor" 
    
    # 定义实验输出路径
    base_output_dir = f"../experiment/{model_name}/YUV420/{width}x{height}/"
    enc_dir = os.path.join(base_output_dir, "enc")
    dec_dir = os.path.join(base_output_dir, "dec_yuv420")
    
    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)
    
    # === VVC 编码器路径设置 ===
    vvc_bin = "../VVC_anchor/bin/EncoderAppStatic_VTM22_2"
    vvc_cfg_main = "../VVC_anchor/cfg/encoder_lowdelay_vtm.cfg" 
    vvc_cfg_input = "../VVC_anchor/cfg/per-sequence/inputyuv420.cfg"

    # 检查编码器是否存在
    if not os.path.exists(vvc_bin):
        print(f"Error: 找不到 VVC 编码器: {vvc_bin}")
        print("请确保您在 source 目录下运行此脚本，且 VVC_anchor 目录结构正确。")
        sys.exit(1)

    # 准备记录结果的数组
    total_bitrate = np.zeros((len(seqlist) + 1, len(qplist)))
    
    # 设定帧率，用于计算 kbps
    fps = 25 
    
    print(f"=== 开始运行传统 VVC 锚点测试 (Direct YUV Input) ===")
    print(f"输入目录: {yuv_sequence_dir}")
    print(f"数据集: {dataset_name}, 帧数: {frames}, 分辨率: {width}x{height}")
    
    # 2. 遍历序列和QP
    for qp_idx, qp in enumerate(qplist):
        for seq_idx, seq in enumerate(tqdm(seqlist, desc=f"QP {qp}")):
            
            # [修改点 2] 直接构造源 YUV 文件路径
            # 注意：这里假设 YUV 文件名格式为 "Dataset_Seq_WxH_25_8bit_420.yuv"
            # 这与 multiMetric_yuv420.py 中的命名逻辑保持一致
            input_yuv_path = os.path.join(
                yuv_sequence_dir, 
                f"{dataset_name}_{seq}_{width}x{height}_25_8bit_420.yuv"
            )
            
            if not os.path.exists(input_yuv_path):
                print(f"Warning: 找不到源文件 {input_yuv_path}，跳过该序列。")
                continue

            # --- B. 构造 VVC 编码命令 ---
            bin_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.bin")
            rec_yuv = os.path.join(dec_dir, f"{dataset_name}_{seq}_{width}x{height}_25fps_8bit_420_qp{qp}.yuv")
            log_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.log")
            
            # 构造命令: 直接使用 input_yuv_path 作为输入
            cmd = (
                f"{vvc_bin} "
                f"-c {vvc_cfg_main} "
                f"-c {vvc_cfg_input} "
                f"-q {qp} "
                f"-f {frames} "
                f"-wdt {width} "
                f"-hgt {height} "
                f"-i {input_yuv_path} "
                f"-b {bin_file} "
                f"-o {rec_yuv} "
                f"> {log_file} 2>&1"
            )
            
            # 执行编码
            start_time = time.time()
            ret = os.system(cmd)
            
            if ret != 0:
                print(f"\nError encoding {seq} at QP {qp}. Check log: {log_file}")
                # 打印日志的最后几行以便调试
                os.system(f"tail -n 5 {log_file}")
                continue
                
            # --- C. 计算码率 (kbps) ---
            bits = get_file_size_in_bits(bin_file)
            
            # [修改点 3] 计算 kbps
            # kbps = (Total Bits * FPS) / (Frames * 1000)
            bitrate_kbps = (bits * fps) / (frames * 1000)
            
            total_bitrate[seq_idx][qp_idx] = bitrate_kbps
            
    # 3. 保存结果
    # 计算平均值
    for q in range(len(qplist)):
        valid_seqs = total_bitrate[:-1, q]
        if np.any(valid_seqs > 0):
            total_bitrate[-1][q] = np.mean(valid_seqs[valid_seqs > 0])
        
    eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 保存文件，命名保持 result_bits 以兼容后续脚本，但内容已是 kbps
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_bits.txt"), total_bitrate, fmt='%.5f')
    
    print(f"\n所有编码完成。")
    print(f"码率结果 (kbps): {os.path.join(eval_dir, f'{dataset_name}_result_bits.txt')}")