import torch
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import cv2
import os
import json


class SmartKeyframeSelector:
    """
    智能关键帧选择器 - 编码前一次性扫描，基于关键点距离选择最优关键帧
    特点：
    1. 编码前一次性扫描所有帧
    2. 基于关键点距离选择最优关键帧
    3. 支持自定义关键帧数量 (num_keyframes)
    """
    
    def __init__(self, analysis_model, device='cuda', num_keyframes=1):
        """
        Args:
            analysis_model: 关键点提取模型
            device: 运行设备
            num_keyframes: 关键帧数量，默认为1
        """
        self.analysis_model = analysis_model
        self.device = device
        self.num_keyframes = num_keyframes
        
    def extract_keypoints(self, listR, listG, listB, width, height, frames):
        """
        一次性扫描所有帧，提取关键点
        
        Returns:
            all_keypoints: list of dict, 每帧的关键点信息
        """
        all_keypoints = []
        
        print(f"[KeyframeSelector] 扫描 {frames} 帧提取关键点...")
        
        for frame_idx in tqdm(range(frames), desc="提取关键点"):
            # 合并 RGB 通道
            frame = cv2.merge([listR[frame_idx], listG[frame_idx], listB[frame_idx]])
            frame = resize(frame, (width, height))[..., :3]
            frame_tensor = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            frame_tensor = frame_tensor.to(self.device)
            
            with torch.no_grad():
                kp_output = self.analysis_model(frame_tensor)
                
            # 提取变换后的关键点
            kp_transformed = self._get_transformed_kp(kp_output)
            all_keypoints.append({
                'frame_idx': frame_idx,
                'kp': kp_transformed.cpu().numpy(),  # (1, N, 3)
            })
            
        return all_keypoints
    
    def _get_transformed_kp(self, motion_output):
        """获取变换后的关键点"""
        from GFVC.LivePortrait.liveportrait_coder import get_rotation_matrix, headpose_pred_to_degree
        
        kp = motion_output['kp']
        exp = motion_output['exp']
        scale = motion_output['scale']
        t = motion_output['t']
        pitch = motion_output['pitch']
        yaw = motion_output['yaw']
        roll = motion_output['roll']
        
        # 标准化形状
        if kp.dim() == 2:
            b = kp.shape[0]
            num_kp = kp.shape[1] // 3
            kp = kp.view(b, num_kp, 3)
        if exp.dim() == 2:
            b = exp.shape[0]
            num_kp = exp.shape[1] // 3
            exp = exp.view(b, num_kp, 3)
        if scale.dim() == 1:
            scale = scale.view(-1, 1)
        if t.dim() == 1:
            t = t.view(-1, 3)
            
        # 转换姿态角度
        for k in ('pitch', 'yaw', 'roll'):
            v = locals()[k]
            if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[-1] == 66:
                locals()[k] = headpose_pred_to_degree(v)
            elif isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[-1] == 1:
                locals()[k] = v.view(-1)
        
        # 重新获取可能已修改的变量
        pitch = locals()['pitch']
        yaw = locals()['yaw']
        roll = locals()['roll']
        
        # 计算旋转矩阵并变换关键点
        rot_mat = get_rotation_matrix(pitch, yaw, roll)
        kp_transformed = kp @ rot_mat + exp
        kp_transformed = kp_transformed * scale.unsqueeze(-1)
        kp_transformed[:, :, 0:2] = kp_transformed[:, :, 0:2] + t[:, None, 0:2]
        
        return kp_transformed
    
    def compute_keypoint_distance(self, kp1, kp2):
        """
        计算两组关键点之间的欧氏距离
        """
        if kp1.ndim == 3:
            kp1 = kp1.reshape(-1)
        if kp2.ndim == 3:
            kp2 = kp2.reshape(-1)
            
        distance = np.linalg.norm(kp1 - kp2)
        return distance
    
    def select_best_keyframes(self, all_keypoints):
        """
        基于最大分散度选择多个关键帧
        
        策略：
        1. 先选择全局"中心"帧（平均距离最小）
        2. 然后贪心选择与其他已选帧距离最大的帧
        
        Returns:
            keyframe_indices: 关键帧索引列表
            distances_matrix: 各帧之间的距离矩阵
        """
        n_frames = len(all_keypoints)
        
        if n_frames <= self.num_keyframes:
            return list(range(n_frames)), None
            
        # 只采样部分帧进行距离计算以加速
        sample_step = max(1, n_frames // 50)  # 最多采样50帧
        sample_indices = list(range(0, n_frames, sample_step))
        if sample_indices[-1] != n_frames - 1:
            sample_indices.append(n_frames - 1)
        
        n_samples = len(sample_indices)
        sample_distances = np.zeros((n_samples, n_samples))
        
        print(f"[KeyframeSelector] 计算 {n_samples} 个采样帧的距离矩阵...")
        
        # 计算采样帧之间的距离
        for i_idx, i in enumerate(sample_indices):
            for j_idx in range(i_idx + 1, n_samples):
                j = sample_indices[j_idx]
                dist = self.compute_keypoint_distance(
                    all_keypoints[i]['kp'], 
                    all_keypoints[j]['kp']
                )
                sample_distances[i_idx, j_idx] = dist
                sample_distances[j_idx, i_idx] = dist
        
        # 选择关键帧
        selected_indices = []
        
        # 第一个关键帧：选择"中心"帧（平均距离最小）
        avg_distances = np.mean(sample_distances, axis=1)
        first_idx = int(np.argmin(avg_distances))
        selected_indices.append(sample_indices[first_idx])
        
        # 后续关键帧：贪心选择距离已选帧最远的帧
        for _ in range(1, min(self.num_keyframes, n_samples)):
            max_min_dist = -1
            best_idx = -1
            
            for i_idx, i in enumerate(sample_indices):
                if i in selected_indices:
                    continue
                # 计算该帧到所有已选帧的最小距离
                selected_sample_indices = [sample_indices.index(idx) for idx in selected_indices if idx in sample_indices]
                if selected_sample_indices:
                    min_dist = min(sample_distances[i_idx, j] for j in selected_sample_indices)
                else:
                    min_dist = 0
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
        
        # 确保关键帧按顺序排列
        selected_indices = sorted(selected_indices)
        
        print(f"[KeyframeSelector] 选择 {len(selected_indices)} 个关键帧: {selected_indices}")
        
        return selected_indices, sample_distances
    
    def select_keyframe(self, listR, listG, listB, width, height, frames):
        """
        主入口：扫描并选择最优关键帧(单帧，向后兼容)
        
        Returns:
            best_idx: 最优关键帧索引
        """
        # 步骤1: 提取所有关键点
        all_keypoints = self.extract_keypoints(listR, listG, listB, width, height, frames)
        
        # 步骤2: 选择最优关键帧
        if self.num_keyframes == 1:
            best_idx, _ = self.select_best_keyframes(all_keypoints)
            return best_idx
        else:
            keyframe_indices, _ = self.select_best_keyframes(all_keypoints)
            return keyframe_indices


def save_keyframe_metadata(keyframe_idx, output_path):
    """保存关键帧索引到元数据文件
    
    支持单关键帧(int)或多关键帧(list)
    """
    # 统一处理为列表格式
    if isinstance(keyframe_idx, int):
        keyframe_list = [keyframe_idx]
        num_keyframes = 1
    else:
        keyframe_list = list(keyframe_idx)
        num_keyframes = len(keyframe_list)
    
    metadata = {
        'keyframe_idx': keyframe_list[0] if num_keyframes == 1 else keyframe_list[0],  # 主关键帧
        'keyframe_list': keyframe_list,  # 所有关键帧
        'num_keyframes': num_keyframes,
        'method': 'smart_keyframe_selector',
        'description': f'基于最大分散度选择的{num_keyframes}个关键帧'
    }
    metadata_path = os.path.join(output_path, 'keyframe_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[KeyframeSelector] 元数据已保存: {num_keyframes}个关键帧 {keyframe_list} -> {metadata_path}")


def load_keyframe_metadata(output_path):
    """从元数据文件加载关键帧索引
    
    Returns:
        keyframe_idx: 主关键帧索引(单个int)
        keyframe_list: 所有关键帧索引列表
        num_keyframes: 关键帧数量
    """
    metadata_path = os.path.join(output_path, 'keyframe_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 优先使用新的列表格式
        if 'keyframe_list' in metadata:
            keyframe_list = metadata['keyframe_list']
            num_keyframes = metadata.get('num_keyframes', len(keyframe_list))
            return keyframe_list[0], keyframe_list, num_keyframes
        else:
            # 向后兼容旧格式
            keyframe_idx = metadata.get('keyframe_idx', 0)
            return keyframe_idx, [keyframe_idx], 1
    
    return 0, [0], 1
