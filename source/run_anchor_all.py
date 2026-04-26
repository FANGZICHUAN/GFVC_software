import os
import sys
import numpy as np
import argparse
import time
import yaml
import cv2
import torch
from tqdm import tqdm
from scipy.signal import convolve2d

# 添加项目路径以导入 LPIPS/DISTS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluate'))
from eval_utils.metrics import LPIPSvgg, DISTS

# 复用项目现有的工具函数读取配置
from GFVC.utils import read_config_file

############################Configuration########################################

# VVC 编码器路径配置
VVC_BIN = "../VVC_anchor/bin/EncoderAppStatic_VTM22_2"
VVC_CFG_MAIN = "../VVC_anchor/cfg/encoder_lowdelay_vtm.cfg"
VVC_CFG_INPUT = "../VVC_anchor/cfg/per-sequence/inputyuv420.cfg"

# 执行开关
ENCODE = True
EVALUATE = True
SUMMARY_RESULTS = True

# 设备配置（用于 LPIPS/DISTS）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

###########################################################################################################

MAX_VALUE = 255  # 8-bit


class VideoCaptureYUV(object):
    """YUV420 文件读取器"""
    def __init__(self, filename, resolution, bitdepth):
        self.file = open(filename, 'rb')
        self.width, self.height = resolution
        self.uv_width = self.width // 2
        self.uv_height = self.height // 2
        self.bitdepth = bitdepth

    def read_frame(self):
        """读取一帧 YUV420"""
        Y = self.read_channel(self.height, self.width)
        U = self.read_channel(self.uv_height, self.uv_width)
        V = self.read_channel(self.uv_height, self.uv_width)
        if Y is None or U is None or V is None:
            return None
        return Y, U, V

    def read_channel(self, height, width):
        """读取单个通道"""
        channel_len = height * width
        raw = self.file.read(channel_len)
        if len(raw) < channel_len:
            return None
        channel = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return channel.astype(np.float32)

    def close(self):
        self.file.close()


def get_file_size_in_bits(file_path):
    """获取文件大小（比特）"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) * 8
    return 0


def calculate_psnr(original, encoded, resolution, frames, original_bitdepth, encoded_bitdepth):
    """计算 YUV-PSNR（YUV加权）"""
    original_video = VideoCaptureYUV(original, resolution, original_bitdepth)
    encoded_video = VideoCaptureYUV(encoded, resolution, encoded_bitdepth)

    psnr_y_array, psnr_u_array, psnr_v_array, psnr_yuv_array = [], [], [], []

    for _ in range(frames):
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
        mse_yuv = (4 * mse_y + mse_u + mse_v) / 6

        # PSNR
        psnr_y = 10 * np.log10((MAX_VALUE ** 2) / (mse_y + 1e-10))
        psnr_u = 10 * np.log10((MAX_VALUE ** 2) / (mse_u + 1e-10))
        psnr_v = 10 * np.log10((MAX_VALUE ** 2) / (mse_v + 1e-10))
        psnr_yuv = 10 * np.log10((MAX_VALUE ** 2) / (mse_yuv + 1e-10))

        psnr_y_array.append(psnr_y)
        psnr_u_array.append(psnr_u)
        psnr_v_array.append(psnr_v)
        psnr_yuv_array.append(psnr_yuv)

    original_video.close()
    encoded_video.close()

    return np.mean(psnr_y_array), np.mean(psnr_u_array), np.mean(psnr_v_array), np.mean(psnr_yuv_array)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """生成高斯核"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    """2D滤波"""
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    """计算单通道 SSIM"""
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(window)

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigma12 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def calculate_ssim_yuv(original, encoded, resolution, frames):
    """计算 YUV-SSIM（YUV加权）"""
    width, height = resolution
    original_video = VideoCaptureYUV(original, resolution, 8)
    encoded_video = VideoCaptureYUV(encoded, resolution, 8)

    ssim_y_array, ssim_u_array, ssim_v_array, ssim_yuv_array = [], [], [], []

    for _ in range(frames):
        out_org = original_video.read_frame()
        out_enc = encoded_video.read_frame()
        if out_org is None or out_enc is None:
            break

        original_y, original_u, original_v = out_org
        encoded_y, encoded_u, encoded_v = out_enc

        # 上采样 UV
        u_up = cv2.resize(original_u, (width, height), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize(original_v, (width, height), interpolation=cv2.INTER_LINEAR)
        enc_u_up = cv2.resize(encoded_u, (width, height), interpolation=cv2.INTER_LINEAR)
        enc_v_up = cv2.resize(encoded_v, (width, height), interpolation=cv2.INTER_LINEAR)

        ssim_y = compute_ssim(original_y, encoded_y)
        ssim_u = compute_ssim(u_up, enc_u_up)
        ssim_v = compute_ssim(v_up, enc_v_up)
        ssim_yuv = (4 * ssim_y + ssim_u + ssim_v) / 6

        ssim_y_array.append(ssim_y)
        ssim_u_array.append(ssim_u)
        ssim_v_array.append(ssim_v)
        ssim_yuv_array.append(ssim_yuv)

    original_video.close()
    encoded_video.close()

    return np.mean(ssim_y_array), np.mean(ssim_u_array), np.mean(ssim_v_array), np.mean(ssim_yuv_array)


def yuv_to_rgb_frame(y, u, v, width, height):
    """将单帧 YUV420 转换为 RGB，使用 I420 平面格式"""
    # 构建 I420 平面格式 (H*3//2, W)
    i420 = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    i420[0:height, :] = y.astype(np.uint8)
    i420[height:height + height // 2, 0:width // 2] = u.astype(np.uint8)
    i420[height:height + height // 2, width // 2:width] = v.astype(np.uint8)
    # 使用正确的颜色空间转换代码
    rgb = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
    return rgb


def calculate_lpips_dists(original, encoded, resolution, frames, lpips_model, dists_model, device='cuda'):
    """计算 LPIPS 和 DISTS"""
    width, height = resolution
    original_video = VideoCaptureYUV(original, resolution, 8)
    encoded_video = VideoCaptureYUV(encoded, resolution, 8)

    lpips_list, dists_list = [], []

    for _ in range(frames):
        out_org = original_video.read_frame()
        out_enc = encoded_video.read_frame()
        if out_org is None or out_enc is None:
            break

        original_y, original_u, original_v = out_org
        encoded_y, encoded_u, encoded_v = out_enc

        # 转 RGB
        orig_rgb = yuv_to_rgb_frame(original_y, original_u, original_v, width, height)
        enc_rgb = yuv_to_rgb_frame(encoded_y, encoded_u, encoded_v, width, height)

        # 转 tensor [1, 3, H, W]
        orig_tensor = torch.from_numpy(orig_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        enc_tensor = torch.from_numpy(enc_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            lpips_val = lpips_model(orig_tensor, enc_tensor, as_loss=False).item()
            dists_val = dists_model(orig_tensor, enc_tensor, as_loss=False).item()

        lpips_list.append(lpips_val)
        dists_list.append(dists_val)

    original_video.close()
    encoded_video.close()

    return np.mean(lpips_list), np.mean(dists_list)


def encode_phase(args, seqlist, frames, width, height, qplist, dataset_name, yuv_sequence_dir):
    """编码阶段：使用 VTM 编码所有序列和 QP 点"""
    print(f"=== 开始 VVC 编码阶段 ===")

    model_name = "VVC_Anchor"
    base_output_dir = os.path.join("..", "experiment", model_name, "YUV420", f"{width}x{height}")
    enc_dir = os.path.join(base_output_dir, "enc")
    dec_dir = os.path.join(base_output_dir, "dec_yuv420")

    os.makedirs(enc_dir, exist_ok=True)
    os.makedirs(dec_dir, exist_ok=True)

    # 检查编码器
    if not os.path.exists(VVC_BIN):
        print(f"Error: 找不到 VVC 编码器: {VVC_BIN}")
        sys.exit(1)

    # 码率数组
    total_bitrate = np.zeros((len(seqlist) + 1, len(qplist)))
    fps = 25

    for qp_idx, qp in enumerate(qplist):
        for seq_idx, seq in enumerate(tqdm(seqlist, desc=f"Encoding QP {qp}")):
            input_yuv_path = os.path.join(
                yuv_sequence_dir,
                f"{dataset_name}_{seq}_{width}x{height}_25_8bit_420.yuv"
            )

            if not os.path.exists(input_yuv_path):
                print(f"Warning: 找不到源文件 {input_yuv_path}，跳过该序列。")
                continue

            bin_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.bin")
            rec_yuv = os.path.join(dec_dir, f"{dataset_name}_{seq}_{width}x{height}_25fps_8bit_420_qp{qp}.yuv")
            log_file = os.path.join(enc_dir, f"{dataset_name}_{seq}_qp{qp}.log")

            cmd = (
                f"{VVC_BIN} "
                f"-c {VVC_CFG_MAIN} "
                f"-c {VVC_CFG_INPUT} "
                f"-q {qp} "
                f"-f {frames} "
                f"-wdt {width} "
                f"-hgt {height} "
                f"-i {input_yuv_path} "
                f"-b {bin_file} "
                f"-o {rec_yuv} "
                f"> {log_file} 2>&1"
            )

            ret = os.system(cmd)
            if ret != 0:
                print(f"\nError encoding {seq} at QP {qp}. Check log: {log_file}")
                os.system(f"tail -n 5 {log_file}")
                continue

            # 计算码率
            bits = get_file_size_in_bits(bin_file)
            bitrate_kbps = (bits * fps) / (frames * 1000)
            total_bitrate[seq_idx][qp_idx] = bitrate_kbps

    # 计算平均值
    for q in range(len(qplist)):
        valid_seqs = total_bitrate[:-1, q]
        if np.any(valid_seqs > 0):
            total_bitrate[-1][q] = np.mean(valid_seqs[valid_seqs > 0])

    # 保存码率结果
    eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
    os.makedirs(eval_dir, exist_ok=True)
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_bits.txt"), total_bitrate, fmt='%.5f')

    print(f"\n编码完成！码率结果: {os.path.join(eval_dir, f'{dataset_name}_result_bits.txt')}")
    return total_bitrate, base_output_dir, enc_dir, dec_dir


def evaluate_phase(args, seqlist, frames, width, height, qplist, dataset_name, yuv_sequence_dir, dec_dir):
    """评估阶段：计算质量指标"""
    print(f"\n=== 开始质量评估阶段 ===")

    model_name = "VVC_Anchor"
    base_output_dir = os.path.join("..", "experiment", model_name, "YUV420", f"{width}x{height}")
    eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
    os.makedirs(eval_dir, exist_ok=True)

    resolution = (width, height)

    # 初始化指标数组
    totalResult_PSNR = np.zeros((len(seqlist) + 1, len(qplist)))
    totalResult_SSIM = np.zeros((len(seqlist) + 1, len(qplist)))
    totalResult_LPIPS = np.zeros((len(seqlist) + 1, len(qplist)))
    totalResult_DISTS = np.zeros((len(seqlist) + 1, len(qplist)))

    # 初始化深度学习模型
    print("Loading LPIPS and DISTS models...")
    lpips_model = LPIPSvgg(device=DEVICE).to(DEVICE)
    dists_model = DISTS(device=DEVICE).to(DEVICE)
    print("Models loaded.")

    for qp_idx, qp in enumerate(qplist):
        for seq_idx, seq in enumerate(tqdm(seqlist, desc=f"Evaluating QP {qp}")):
            # 寻找参考文件
            original_yuv = os.path.join(
                yuv_sequence_dir,
                f"{dataset_name}_{seq}_{width}x{height}_25_8bit_420.yuv"
            )

            decoded_yuv = os.path.join(dec_dir, f"{dataset_name}_{seq}_{width}x{height}_25fps_8bit_420_qp{qp}.yuv")

            if not os.path.exists(original_yuv):
                print(f"Error: 找不到参考文件 {original_yuv}")
                continue
            if not os.path.exists(decoded_yuv):
                print(f"Error: 找不到解码文件 {decoded_yuv}")
                continue

            # 计算 PSNR
            psnr_y, psnr_u, psnr_v, psnr_yuv = calculate_psnr(
                original_yuv, decoded_yuv, resolution, frames, 8, 8
            )
            totalResult_PSNR[seq_idx][qp_idx] = psnr_yuv

            # 计算 SSIM
            ssim_y, ssim_u, ssim_v, ssim_yuv = calculate_ssim_yuv(
                original_yuv, decoded_yuv, resolution, frames
            )
            totalResult_SSIM[seq_idx][qp_idx] = ssim_yuv

            # 计算 LPIPS 和 DISTS
            lpips_val, dists_val = calculate_lpips_dists(
                original_yuv, decoded_yuv, resolution, frames, lpips_model, dists_model, DEVICE
            )
            totalResult_LPIPS[seq_idx][qp_idx] = lpips_val
            totalResult_DISTS[seq_idx][qp_idx] = dists_val

    # 计算平均值
    for q in range(len(qplist)):
        for arr in [totalResult_PSNR, totalResult_SSIM, totalResult_LPIPS, totalResult_DISTS]:
            valid = arr[:-1, q]
            if np.any(valid > 0):
                arr[-1][q] = np.mean(valid[valid > 0])

    # 保存所有结果
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_psnr.txt"), totalResult_PSNR, fmt='%.5f')
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_ssim.txt"), totalResult_SSIM, fmt='%.5f')
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_lpips.txt"), totalResult_LPIPS, fmt='%.5f')
    np.savetxt(os.path.join(eval_dir, f"{dataset_name}_result_dists.txt"), totalResult_DISTS, fmt='%.5f')

    print(f"\n评估完成！结果已保存至: {eval_dir}")
    return totalResult_PSNR, totalResult_SSIM, totalResult_LPIPS, totalResult_DISTS, eval_dir


def summary_phase(args, seqlist, qplist, dataset_name, bitrate_arr, psnr_arr, ssim_arr, lpips_arr, dists_arr, eval_dir):
    """汇总阶段：生成 GFVC 格式的汇总文件"""
    print(f"\n=== 开始生成汇总结果 ===")

    model_name = "VVC_Anchor"
    width = args['width']
    height = args['height']
    tim = time.localtime()

    base_dir = os.path.join("..", "experiment", model_name, "YUV420", f"{width}x{height}")
    summary_path = os.path.join(
        eval_dir,
        f"{tim[0]}-{str(tim[1]).zfill(2)}-{str(tim[2]).zfill(2)}T"
        f"{str(tim[3]).zfill(2)}-{str(tim[4]).zfill(2)}-{str(tim[5]).zfill(2)}-"
        f"{model_name}-EvaluationSummary.txt"
    )

    with open(summary_path, 'w') as f:
        # 码率
        f.write(f'########## {model_name}_{dataset_name}_Bitrate ##########' + '\n')
        for q in range(len(qplist)):
            f.write(f"{bitrate_arr[-1][q]:.5f}\n")

        # PSNR
        f.write(f'########## {model_name}_{dataset_name}_YUV420_psnr ##########' + '\n')
        for q in range(len(qplist)):
            f.write(f"{psnr_arr[-1][q]:.5f}\n")

        # SSIM
        f.write(f'########## {model_name}_{dataset_name}_YUV420_ssim ##########' + '\n')
        for q in range(len(qplist)):
            f.write(f"{ssim_arr[-1][q]:.5f}\n")

        # LPIPS
        f.write(f'########## {model_name}_{dataset_name}_YUV420_lpips ##########' + '\n')
        for q in range(len(qplist)):
            # LPIPS 越小越好，但汇总格式通常保持原始值
            f.write(f"{lpips_arr[-1][q]:.5f}\n")

        # DISTS
        f.write(f'########## {model_name}_{dataset_name}_YUV420_dists ##########' + '\n')
        for q in range(len(qplist)):
            f.write(f"{dists_arr[-1][q]:.5f}\n")

    print(f"汇总文件已保存至: {summary_path}")
    return summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../config/anchor.yaml', type=str, help="Path to codec configuration file")
    opt = parser.parse_args()

    # 读取配置
    args = read_config_file(opt.config)
    seqlist = args['seq_list']
    frames = args['num_frames']
    width = args['width']
    height = args['height']
    qplist = args['qp_list']
    dataset_name = args['dataset']
    yuv_sequence_dir = args['yuv_sequence_dir']

    print(f"=== VVC Anchor 全流程测试 ===")
    print(f"数据集: {dataset_name}")
    print(f"序列: {seqlist}")
    print(f"分辨率: {width}x{height}")
    print(f"帧数: {frames}")
    print(f"QP列表: {qplist}")
    print(f"设备: {DEVICE}")

    # 阶段 1: 编码
    if ENCODE:
        bitrate_arr, base_output_dir, enc_dir, dec_dir = encode_phase(
            args, seqlist, frames, width, height, qplist, dataset_name, yuv_sequence_dir
        )
    else:
        base_output_dir = os.path.join("..", "experiment", "VVC_Anchor", "YUV420", f"{width}x{height}")
        enc_dir = os.path.join(base_output_dir, "enc")
        dec_dir = os.path.join(base_output_dir, "dec_yuv420")
        eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
        bitrate_arr = np.loadtxt(os.path.join(eval_dir, f"{dataset_name}_result_bits.txt"))

    # 阶段 2: 评估
    if EVALUATE:
        psnr_arr, ssim_arr, lpips_arr, dists_arr, eval_dir = evaluate_phase(
            args, seqlist, frames, width, height, qplist, dataset_name, yuv_sequence_dir, dec_dir
        )
    else:
        eval_dir = os.path.join(base_output_dir, "evaluation-YUV420")
        psnr_arr = np.loadtxt(os.path.join(eval_dir, f"{dataset_name}_result_psnr.txt"))
        ssim_arr = np.loadtxt(os.path.join(eval_dir, f"{dataset_name}_result_ssim.txt"))
        lpips_arr = np.loadtxt(os.path.join(eval_dir, f"{dataset_name}_result_lpips.txt"))
        dists_arr = np.loadtxt(os.path.join(eval_dir, f"{dataset_name}_result_dists.txt"))

    # 阶段 3: 汇总
    if SUMMARY_RESULTS:
        summary_path = summary_phase(
            args, seqlist, qplist, dataset_name, bitrate_arr, psnr_arr, ssim_arr, lpips_arr, dists_arr, eval_dir
        )

    print("\n=== 所有阶段完成！===")
