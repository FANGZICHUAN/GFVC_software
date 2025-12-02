import numpy as np
import os
import argparse
import yaml
import glob
from tqdm import tqdm

# 这一段是为了复用 calculate_psnr 和 calculate_ssim 函数
# 我们直接把它们从 multiMetric_yuv420.py 搬过来，或者导入
# 为了脚本独立性，这里直接包含核心计算逻辑

MAX_VALUE = 255 # 8-bit

class VideoCaptureYUV(object):
    def __init__(self, filename, resolution, bitdepth):
        self.file = open(filename, 'rb')
        self.width, self.height = resolution
        self.uv_width = self.width // 2
        self.uv_height = self.height // 2
        self.bitdepth = bitdepth

    def read_frame(self):
        # 读取 Y
        Y = self.read_channel(self.height, self.width)
        # 读取 U, V
        U = self.read_channel(self.uv_height, self.uv_width)
        V = self.read_channel(self.uv_height, self.uv_width)
        return Y, U, V

    def read_channel(self, height, width):
        channel_len = height * width
        shape = (height, width)
        raw = self.file.read(channel_len)
        if len(raw) < channel_len:
            return None # End of file
        channel_8bits = np.frombuffer(raw, dtype=np.uint8)
        channel = np.array(channel_8bits, dtype=np.float32)
        channel = channel.reshape(shape)
        return channel

    def close(self):
        self.file.close()

def calculate_psnr(original, encoded, resolution, frames, original_bitdepth, encoded_bitdepth):
    original_video = VideoCaptureYUV(original, resolution, original_bitdepth)
    encoded_video = VideoCaptureYUV(encoded, resolution, encoded_bitdepth)

    psnr_y_array = []
    psnr_u_array = []
    psnr_v_array = []
    psnr_yuv_array = []

    for frame in range(frames):
        out_org = original_video.read_frame()
        out_enc = encoded_video.read_frame()
        
        if out_org is None or out_enc is None:
            break
            
        original_y, original_u, original_v = out_org
        encoded_y, encoded_u, encoded_v = out_enc

        # MSE
        mse_y = np.mean((original_y - encoded_y) ** 2)
        mse_u = np.mean((original_u - encoded_u) ** 2)
        mse_v = np.mean((original_v - encoded_v) ** 2)
        
        # Weighted MSE for YUV PSNR (4:1:1 weight is common in VVC/HEVC papers)
        mse_yuv_frame = (4 * mse_y + mse_u + mse_v) / 6

        # PSNR
        psnr_y = 10 * np.log10((MAX_VALUE ** 2) / (mse_y + 1e-10))
        psnr_u = 10 * np.log10((MAX_VALUE ** 2) / (mse_u + 1e-10))
        psnr_v = 10 * np.log10((MAX_VALUE ** 2) / (mse_v + 1e-10))
        psnr_yuv = 10 * np.log10((MAX_VALUE ** 2) / (mse_yuv_frame + 1e-10))

        psnr_y_array.append(psnr_y)
        psnr_u_array.append(psnr_u)
        psnr_v_array.append(psnr_v)
        psnr_yuv_array.append(psnr_yuv)

    original_video.close()
    encoded_video.close()

    return np.mean(psnr_y_array), np.mean(psnr_u_array), np.mean(psnr_v_array), np.mean(psnr_yuv_array)

def read_config_file(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../config/anchor.yaml', type=str)
    opt = parser.parse_args()
    args = read_config_file(opt.config)

    seqlist = args['seq_list']
    frames = args['num_frames']
    width = args['width']
    height = args['height']
    qplist = args['qp_list']
    dataset_name = args['dataset']
    model_name = "VVC_Anchor" # 强制指定，因为我们知道 run_anchor.py 用的是这个名字

    # 路径配置 (必须与 run_anchor.py 一致)
    base_output_dir = f"../experiment/{model_name}/YUV420/{width}x{height}/"
    dec_dir = os.path.join(base_output_dir, "dec_yuv420")
    eval_output_dir = os.path.join(base_output_dir, "evaluation-YUV420")
    os.makedirs(eval_output_dir, exist_ok=True)

    resolution = (width, height)

    totalResult_PSNR = np.zeros((len(seqlist) + 1, len(qplist)))
    
    print(f"=== 开始评估 VVC Anchor ===")
    print(f"结果目录: {dec_dir}")

    for qp_idx, qp in enumerate(qplist):
        for seq_idx, seq in enumerate(tqdm(seqlist, desc=f"Evaluating QP {qp}")):
            
            # 1. 寻找参考文件 (Ground Truth)
            # 优先使用 run_anchor.py 生成的 _input.yuv，保证完全对齐
            original_yuv = os.path.join(base_output_dir, f"{dataset_name}_{seq}_input.yuv")
            
            # 如果找不到，尝试去原始数据集找 (备选方案)
            if not os.path.exists(original_yuv):
                # 假设命名格式
                original_yuv = args['yuv_sequence_dir'] + f"{dataset_name}_{seq}_{width}x{height}_25_8bit_420.yuv"
            
            # 2. 寻找解码文件 (Reconstructed)
            decoded_yuv = os.path.join(dec_dir, f"{dataset_name}_{seq}_{width}x{height}_25fps_8bit_420_qp{qp}.yuv")

            if not os.path.exists(original_yuv):
                print(f"Error: 找不到参考文件 {original_yuv}")
                continue
            if not os.path.exists(decoded_yuv):
                print(f"Error: 找不到解码文件 {decoded_yuv}")
                continue

            # 3. 计算指标
            psnr_y, psnr_u, psnr_v, psnr_yuv = calculate_psnr(
                original_yuv, decoded_yuv, resolution, frames, 8, 8
            )
            
            # 我们主要记录 YUV-PSNR (或者您可以根据需要改存 Y-PSNR)
            totalResult_PSNR[seq_idx][qp_idx] = psnr_yuv

    # 计算平均值
    for q in range(len(qplist)):
        valid_scores = totalResult_PSNR[:-1, q]
        if np.any(valid_scores > 0):
            totalResult_PSNR[-1][q] = np.mean(valid_scores[valid_scores > 0])

    # 保存结果
    np.savetxt(os.path.join(eval_output_dir, f"{dataset_name}_result_psnr.txt"), totalResult_PSNR, fmt='%.5f')
    print(f"\n评估完成！结果已保存至: {os.path.join(eval_output_dir, f'{dataset_name}_result_psnr.txt')}")