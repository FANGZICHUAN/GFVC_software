import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def parse_evaluation_summary(file_path):
    """解析EvaluationSummary.txt文件"""
    data = {}
    current_key = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('##########'):
                current_key = line.replace('#', '').strip()
                data[current_key] = []
            elif line and current_key:
                try:
                    data[current_key].append(float(line))
                except ValueError:
                    pass
    
    return data

def calculate_bd_rate(rate1, psnr1, rate2, psnr2):
    """
    计算BD-rate (Bjontegaard Delta Rate)
    使用分段线性积分方法，更稳健
    rate1, psnr1: 参考曲线的码率和PSNR
    rate2, psnr2: 测试曲线的码率和PSNR
    返回: BD-rate (%)
    """
    # 确保数据按PSNR排序
    idx1 = np.argsort(psnr1)
    psnr1 = np.array(psnr1)[idx1]
    rate1 = np.array(rate1)[idx1]
    
    idx2 = np.argsort(psnr2)
    psnr2 = np.array(psnr2)[idx2]
    rate2 = np.array(rate2)[idx2]
    
    # 转换为log域
    log_rate1 = np.log(rate1)
    log_rate2 = np.log(rate2)
    
    # 确定PSNR的公共范围
    min_psnr = max(min(psnr1), min(psnr2))
    max_psnr = min(max(psnr1), max(psnr2))
    
    # 检查公共范围是否有效
    if min_psnr >= max_psnr:
        return np.nan
    
    # 使用线性插值（更稳健）
    from scipy.interpolate import interp1d
    
    # 创建插值函数
    f1 = interp1d(psnr1, log_rate1, kind='linear', fill_value='extrapolate')
    f2 = interp1d(psnr2, log_rate2, kind='linear', fill_value='extrapolate')
    
    # 在公共范围内采样
    psnr_samples = np.linspace(min_psnr, max_psnr, 100)
    
    # 计算积分（使用梯形法则）
    log_rate1_samples = f1(psnr_samples)
    log_rate2_samples = f2(psnr_samples)
    
    int1 = np.trapezoid(log_rate1_samples, psnr_samples)
    int2 = np.trapezoid(log_rate2_samples, psnr_samples)
    
    # 计算平均差异
    avg_diff = (int2 - int1) / (max_psnr - min_psnr)
    
    # 限制avg_diff的范围以避免溢出
    if avg_diff > 10:  # exp(10) ≈ 22026
        bd_rate = np.inf
    elif avg_diff < -10:
        bd_rate = -100.0
    else:
        bd_rate = (np.exp(avg_diff) - 1) * 100
    
    return bd_rate

def plot_rd_curves(experiment_dir, output_dir='graph'):
    """
    绘制RD曲线并计算BD-rate
    
    参数:
        experiment_dir: experiment文件夹的路径
        output_dir: 输出图片的文件夹
    """
    models = ['LVPT', 'CFTE', 'TPSMM', 'FV2V']
    colors = {'LVPT': 'blue', 'CFTE': 'red', 'TPSMM': 'green', 'FV2V': 'orange'}
    markers = {'LVPT': 'o', 'CFTE': 's', 'TPSMM': '^', 'FV2V': 'D'}
    
    # 存储所有模型的数据
    all_data = {}
    
    # 读取每个模型的数据
    for model in models:
        pattern = os.path.join(experiment_dir, model, 'Iframe_YUV420/256x256/AE/*-EvaluationSummary.txt')
        files = glob.glob(pattern)
        
        if not files:
            print(f"警告: 未找到 {model} 的评估文件")
            continue
        
        # 使用最新的文件
        file_path = sorted(files)[-1]
        print(f"读取文件: {file_path}")
        
        data = parse_evaluation_summary(file_path)
        
        # 提取需要的指标
        bitrate_key = [k for k in data.keys() if 'Bitrate' in k][0]
        yuv_psnr_key = [k for k in data.keys() if 'YUV420_psnr' in k][0]
        lmd_key = [k for k in data.keys() if 'RGB444_lmd' in k][0]
        
        all_data[model] = {
            'bitrate': np.array(data[bitrate_key]),
            'yuv_psnr': np.array(data[yuv_psnr_key]),
            'lmd': np.array(data[lmd_key])
        }
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形（2个子图）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: YUV420 PSNR vs Bitrate
    ax1 = axes[0]
    for model in models:
        if model not in all_data:
            continue
        
        bitrate = all_data[model]['bitrate']
        yuv_psnr = all_data[model]['yuv_psnr']
        
        ax1.plot(bitrate, yuv_psnr, 
                marker=markers[model], 
                color=colors[model], 
                linewidth=2, 
                markersize=8,
                label=model)
    
    ax1.set_xlabel('Bitrate (kbps)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('YUV420 PSNR (dB)', fontsize=12, fontweight='bold')
    ax1.set_title('Rate-Distortion Curve (YUV420 PSNR)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # 子图2: LMD vs Bitrate
    ax2 = axes[1]
    for model in models:
        if model not in all_data:
            continue
        
        bitrate = all_data[model]['bitrate']
        lmd = all_data[model]['lmd']
        
        ax2.plot(bitrate, lmd, 
                marker=markers[model], 
                color=colors[model], 
                linewidth=2, 
                markersize=8,
                label=model)
    
    ax2.set_xlabel('Bitrate (kbps)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('LMD', fontsize=12, fontweight='bold')
    ax2.set_title('Rate-Distortion Curve (LMD)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.invert_yaxis()  # LMD越小越好，所以反转Y轴
    
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, 'rd_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图片已保存至: {output_path}")
    
    # 打印数据摘要
    print("\n" + "="*70)
    print("数据摘要")
    print("="*70)
    for model in models:
        if model not in all_data:
            continue
        print(f"\n{model}:")
        print(f"  Bitrate: {all_data[model]['bitrate']}")
        print(f"  YUV PSNR: {all_data[model]['yuv_psnr']}")
        print(f"  LMD: {all_data[model]['lmd']}")
    
    # 计算BD-rate (以LVPT为参考)
    print("\n" + "="*70)
    print("BD-Rate 计算结果 (相对于 LVPT, 负值表示性能更好)")
    print("="*70)
    print("\n注意: 如果显示'N/A'，说明两条曲线的质量范围不重叠，无法计算BD-rate")
    
    reference_model = 'LVPT'
    if reference_model in all_data:
        ref_bitrate = all_data[reference_model]['bitrate']
        ref_yuv_psnr = all_data[reference_model]['yuv_psnr']
        ref_lmd = all_data[reference_model]['lmd']
        
        print(f"\n{'模型':<10} {'YUV420 PSNR BD-Rate':<25} {'LMD BD-Rate':<20}")
        print("-"*70)
        
        for model in models:
            if model == reference_model or model not in all_data:
                continue
            
            test_bitrate = all_data[model]['bitrate']
            test_yuv_psnr = all_data[model]['yuv_psnr']
            test_lmd = all_data[model]['lmd']
            
            # 检查PSNR范围是否重叠
            ref_psnr_min, ref_psnr_max = ref_yuv_psnr.min(), ref_yuv_psnr.max()
            test_psnr_min, test_psnr_max = test_yuv_psnr.min(), test_yuv_psnr.max()
            overlap_min = max(ref_psnr_min, test_psnr_min)
            overlap_max = min(ref_psnr_max, test_psnr_max)
            
            # 计算YUV PSNR的BD-rate
            bd_rate_yuv = calculate_bd_rate(ref_bitrate, ref_yuv_psnr, 
                                           test_bitrate, test_yuv_psnr)
            
            # 对于LMD，需要反转（因为LMD越小越好）
            bd_rate_lmd = calculate_bd_rate(ref_bitrate, -ref_lmd, 
                                           test_bitrate, -test_lmd)
            
            # 格式化输出
            if np.isnan(bd_rate_yuv) or overlap_min >= overlap_max:
                yuv_str = "N/A (无重叠)"
            elif np.isinf(bd_rate_yuv):
                yuv_str = ">1000% (性能差异过大)"
            else:
                yuv_str = f"{bd_rate_yuv:>8.2f}%"
            
            if np.isnan(bd_rate_lmd):
                lmd_str = "N/A"
            elif np.isinf(bd_rate_lmd):
                lmd_str = ">1000%"
            else:
                lmd_str = f"{bd_rate_lmd:>8.2f}%"
            
            print(f"{model:<10} {yuv_str:<25} {lmd_str:<20}")
        
        # 添加平均码率比较（用于PSNR范围不重叠的情况）
        print("\n" + "="*70)
        print("平均码率比较 (相对于 LVPT)")
        print("="*70)
        print(f"\n{'模型':<10} {'平均码率 (kbps)':<20} {'相对码率':<20} {'平均YUV PSNR (dB)':<20}")
        print("-"*70)
        
        ref_avg_bitrate = ref_bitrate.mean()
        ref_avg_psnr = ref_yuv_psnr.mean()
        print(f"{'LVPT':<10} {ref_avg_bitrate:>15.2f}     {'100.00%':<20} {ref_avg_psnr:>15.2f}")
        
        for model in models:
            if model == reference_model or model not in all_data:
                continue
            
            test_bitrate = all_data[model]['bitrate']
            test_yuv_psnr = all_data[model]['yuv_psnr']
            
            avg_bitrate = test_bitrate.mean()
            avg_psnr = test_yuv_psnr.mean()
            relative_bitrate = (avg_bitrate / ref_avg_bitrate) * 100
            
            print(f"{model:<10} {avg_bitrate:>15.2f}     {relative_bitrate:>6.2f}%{'':<13} {avg_psnr:>15.2f}")
    
    print("="*70)
    print("\n完成!")
    
    return all_data

if __name__ == '__main__':
    # 设置实验目录路径
    experiment_dir = '../experiment'
    
    # 绘制RD曲线并计算BD-rate
    plot_rd_curves(experiment_dir)
