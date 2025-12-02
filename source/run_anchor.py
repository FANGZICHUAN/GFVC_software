import os
import sys
import numpy as np
import cv2
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
    sequence_dir = args['sequence_dir']
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
    
    # === VVC 编码器路径设置 (关键修改) ===
    # 直接指向可执行文件，避开 shell 脚本的路径问题
    vvc_bin = "../VVC_anchor/bin/EncoderAppStatic_VTM22_2"
    # 使用 Low Delay 配置 (通常用于实时通信场景对比)
    vvc_cfg_main = "../VVC_anchor/cfg/encoder_lowdelay_vtm.cfg" 
    vvc_cfg_input = "../VVC_anchor/cfg/per-sequence/inputyuv420.cfg"

    # 检查编码器是否存在
    if not os.path.exists(vvc_bin):
        print(f"Error: 找不到 VVC 编码器: {vvc_bin}")
        print("请确保您在 source 目录下运行此脚本，且 VVC_anchor 目录结构正确。")
        sys.exit(1)

    # 准备记录结果的数组
    total_bits = np.zeros((len(seqlist) + 1, len(qplist)))
    
    print(f"=== 开始运行传统 VVC 锚点测试 (Direct Binary Call) ===")
    print(f"数据集: {dataset_name}, 帧数: {frames}, 分辨率: {width}x{height}")
    
    # 2. 遍历序列和QP
    for qp_idx, qp in enumerate(qplist):
        for seq_idx, seq in enumerate(tqdm(seqlist, desc=f"QP {qp}")):
            
            # --- A. 准备输入数据 (RGB -> YUV) ---
            original_rgb_path = f"{sequence_dir}{dataset_name}_{seq}_{width}x{height}_25fps_8bit_444.rgb"
            input_yuv_path = os.path.join(base_output_dir, f"{dataset_name}_{seq}_input.yuv")
            
            if not os.path.exists(input_yuv_path):
                file_size = width * height * 3 * frames
                if not os.path.exists(original_rgb_path):
                    print(f"Warning: 找不到源文件 {original_rgb_path}，跳过该序列。")
                    continue

                with open(original_rgb_path, 'rb') as f:
                    with open(input_yuv_path, 'wb') as yuv_file:
                        for _ in range(frames):
                            # 读取 Planar RGB
                            R = np.frombuffer(f.read(width*height), dtype=np.uint8).reshape((height, width))
                            G = np.frombuffer(f.read(width*height), dtype=np.uint8).reshape((height, width))
                            B = np.frombuffer(f.read(width*height), dtype=np.uint8).reshape((height, width))
                            
                            img_rgb = cv2.merge([R, G, B])
                            # 转换为 YUV420
                            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV_I420)
                            img_yuv.tofile(yuv_file)

            # --- B. 构造 VVC 编码命令 ---
            bin_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.bin")
            rec_yuv = os.path.join(dec_dir, f"{dataset_name}_{seq}_{width}x{height}_25fps_8bit_420_qp{qp}.yuv")
            log_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.log")
            
            # 构造命令: 显式传入所有路径
            # -c: 配置文件 (主配置 + 输入格式配置)
            # -q: QP, -f: 帧数, -wdt/-hgt: 宽高
            # -i: 输入, -o: 重建输出, -b: 码流输出
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
                
            # --- C. 记录码率信息 ---
            bits = get_file_size_in_bits(bin_file)
            total_bits[seq_idx][qp_idx] = bits / frames 
            
    # 3. 保存结果
    # 计算平均值
    for q in range(len(qplist)):
        valid_seqs = total_bits[:-1, q]
        # 避免除以零（如果所有序列都失败）
        if np.any(valid_seqs > 0):
            total_bits[-1][q] = np.mean(valid_seqs[valid_seqs > 0])
        
    eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
    os.makedirs(eval_dir, exist_ok=True)
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_bits.txt"), total_bits, fmt='%.5f')
    
    print(f"\n所有编码完成。")
    print(f"码率结果: {os.path.join(eval_dir, f'{dataset_name}_result_bits.txt')}")