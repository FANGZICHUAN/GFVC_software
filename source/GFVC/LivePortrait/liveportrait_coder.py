import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np

# LivePortrait core networks
from GFVC.LivePortrait.modules.motion_extractor import MotionExtractor
from GFVC.LivePortrait.modules.warping_network import WarpingNetwork
from GFVC.LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from GFVC.LivePortrait.modules.spade_generator import SPADEDecoder

# GFVC arithmetic coding modules
from arithmetic.value_encoder import final_encoder_expgolomb
from arithmetic.value_decoder import final_decoder_expgolomb, data_convert_inverse_expgolomb


def headpose_pred_to_degree(pred):
    """Convert MotionExtractor pose logits (B, 66) to degree (B,)"""
    if pred.ndim > 1 and pred.shape[1] == 66:
        device = pred.device
        idx_tensor = torch.arange(66, dtype=torch.float32, device=device)
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred * idx_tensor, dim=1) * 3 - 97.5
        return degree
    return pred


def get_rotation_matrix(pitch_, yaw_, roll_):
    """Build rotation matrix from Euler angles in degrees. Each input is (B,) or (B,1)."""
    PI = np.pi
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device
    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose as in original


class LivePortraitCoder:
    """
    LivePortrait-based Video Codec
    Uses 3D implicit keypoints for face video compression
    """
    def __init__(self, model_config_path, model_checkpoint_path, kp_output_dir=None, q_step=None, device='cuda'):
        self.device = device
        # LivePortrait features typically in range [-1, 1], use 128 as default scale
        self.q_step = q_step if q_step is not None else [128, 128]
        self.kp_output_dir = kp_output_dir

        # Internal state
        self.reference_frame = None
        self.ref_appearance_features = None
        self.kp_reference = None
        self.rec_sem = []
        self.num_kp = 21

        # Skip network init if config is None (decoder-only mode)
        if model_config_path is None:
            self.appearance_extractor = None
            self.motion_extractor = None
            self.warping_network = None
            self.spade_generator = None
            return

        # 1. Load LivePortrait config
        with open(model_config_path) as f:
            config = yaml.safe_load(f)
        self.num_kp = config['model_params']['common_params'].get('num_kp', 21)

        # 2. Initialize LivePortrait core networks
        # model_checkpoint_path is now a directory containing 4 .pth files
        checkpoint_dir = model_checkpoint_path if model_checkpoint_path else './GFVC/LivePortrait/checkpoint/'
        
        # Initialize networks
        # AppearanceFeatureExtractor doesn't need num_kp from common_params
        appearance_params = config['model_params'].get('appearance_params', {})
        self.appearance_extractor = AppearanceFeatureExtractor(
            image_channel=appearance_params.get('image_channel', 3),
            block_expansion=appearance_params.get('block_expansion', 64),
            num_down_blocks=appearance_params.get('num_down_blocks', 4),
            max_features=appearance_params.get('max_features', 512),
            reshape_channel=appearance_params.get('reshape_channel', 32),
            reshape_depth=appearance_params.get('reshape_depth', 16),
            num_resblocks=appearance_params.get('num_resblocks', 2)
        ).to(self.device)
        
        self.motion_extractor = MotionExtractor(
            **config['model_params']['common_params']
        ).to(self.device)
        
        # WarpingNetwork - explicit parameters
        warp_params = config['model_params']['warping_params']
        self.warping_network = WarpingNetwork(
            num_kp=warp_params.get('num_kp', 21),
            block_expansion=warp_params.get('block_expansion', 64),
            max_features=warp_params.get('max_features', 512),
            num_down_blocks=warp_params.get('num_down_blocks', 2),
            reshape_channel=warp_params.get('reshape_channel', 32),
            estimate_occlusion_map=warp_params.get('estimate_occlusion_map', True),
            dense_motion_params=warp_params.get('dense_motion_params', {
                'block_expansion': 32, 'num_blocks': 5, 'max_features': 1024, 'reshape_depth': 16, 'compress': 4
            })
        ).to(self.device)
        
        # SPADEDecoder has its own parameters, don't pass common_params with num_kp
        gen_params = config['model_params'].get('generator_params', {})
        self.spade_generator = SPADEDecoder(
            upscale=gen_params.get('upscale', 2),
            max_features=gen_params.get('max_features', 512),
            block_expansion=gen_params.get('block_expansion', 64),
            out_channels=gen_params.get('out_channels', 64),
            num_down_blocks=gen_params.get('num_down_blocks', 2)
        ).to(self.device)
        
        # 3. Load pretrained weights from separate files
        try:
            self.appearance_extractor.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'appearance_feature_extractor.pth'),
                          map_location=self.device), strict=True
            )
            self.motion_extractor.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'motion_extractor.pth'),
                          map_location=self.device), strict=True
            )
            self.warping_network.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'warping_module.pth'),
                          map_location=self.device), strict=True
            )
            self.spade_generator.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'spade_generator.pth'),
                          map_location=self.device), strict=True
            )
            print("LivePortrait checkpoints loaded successfully")
        except FileNotFoundError as e:
            print(f"Warning: Could not load checkpoint files from {checkpoint_dir}: {e}")
            print("Models initialized with random weights")
            
        self.appearance_extractor.eval()
        self.motion_extractor.eval()
        self.warping_network.eval()
        self.spade_generator.eval()

    def reference_frame_(self, reference):
        """Store I-frame as reference and extract appearance features.
        When networks are not loaded (kp-only arithmetic mode), only cache the frame."""
        self.reference_frame = reference
        if self.appearance_extractor is None:
            self.ref_appearance_features = None
            return
        with torch.no_grad():
            self.ref_appearance_features = self.appearance_extractor(reference)

    def kp_reference_(self, motion_output):
        """Store I-frame transformed keypoints as kp_source for warping.

        kp_source = (kp @ R + exp) * scale + t (only tx, ty applied)
        """
        motion_output = self._normalize_motion_output(motion_output)
        self.kp_reference = self._transform_keypoints(motion_output)
        self.kp_reference_full = motion_output

    def _transform_keypoints(self, motion_output):
        """Apply LivePortrait Eqn.2: kp_t = (kp @ R + exp) * scale + t[:2]"""
        kp = motion_output['kp']            # (B, N, 3)
        exp = motion_output['exp']          # (B, N, 3)
        scale = motion_output['scale']      # (B, 1)
        t = motion_output['t']              # (B, 3)
        pitch = motion_output['pitch']      # (B,) in degrees
        yaw = motion_output['yaw']
        roll = motion_output['roll']

        rot_mat = get_rotation_matrix(pitch, yaw, roll)  # (B, 3, 3)
        kp_transformed = kp @ rot_mat + exp
        kp_transformed = kp_transformed * scale.unsqueeze(-1)
        kp_transformed[:, :, 0:2] = kp_transformed[:, :, 0:2] + t[:, None, 0:2]
        return kp_transformed

    def _normalize_motion_output(self, motion_output):
        """Normalize shapes and convert pitch/yaw/roll logits to degrees."""
        out = dict(motion_output)

        if out['kp'].dim() == 2:
            b = out['kp'].shape[0]
            num_kp = out['kp'].shape[1] // 3
            out['kp'] = out['kp'].view(b, num_kp, 3)

        if out['exp'].dim() == 2:
            b = out['exp'].shape[0]
            num_kp = out['exp'].shape[1] // 3
            out['exp'] = out['exp'].view(b, num_kp, 3)

        if out['scale'].dim() == 1:
            out['scale'] = out['scale'].view(-1, 1)

        if out['t'].dim() == 1:
            out['t'] = out['t'].view(-1, 3)

        # Convert pose logits (B, 66) to degrees (B,) if needed
        for k in ('pitch', 'yaw', 'roll'):
            if k in out and out[k] is not None:
                v = out[k]
                if isinstance(v, torch.Tensor) and v.ndim > 1 and v.shape[-1] == 66:
                    out[k] = headpose_pred_to_degree(v)
                elif isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[-1] == 1:
                    out[k] = v.view(-1)

        return out

    def analysis_model(self, frame):
        """Encoder: Extract motion features using MotionExtractor
        
        Returns dict with:
        - kp: canonical keypoints (B, num_kp, 3)
        - exp: expression deformation (B, num_kp, 3)
        - scale: scale factor (B, 1)
        - t: translation (B, 3)
        """
        with torch.no_grad():
            motion_output = self.motion_extractor(frame)
        return self._normalize_motion_output(motion_output)

    def get_kp_list(self, motion_output, frame_idx, save=False):
        """Extract parameter list: [kp(63), exp(63), scale(1), t(3), pitch, yaw, roll] = 133 values"""
        motion_output = self._normalize_motion_output(motion_output)

        if frame_idx == 0:
            self.kp_reference_(motion_output)

        kp_flat = motion_output['kp'].flatten().cpu().numpy().tolist()
        exp_flat = motion_output['exp'].flatten().cpu().numpy().tolist()
        scale_flat = motion_output['scale'].flatten().cpu().numpy().tolist()
        t_flat = motion_output['t'].flatten().cpu().numpy().tolist()
        pitch_v = float(motion_output['pitch'].flatten()[0].item())
        yaw_v = float(motion_output['yaw'].flatten()[0].item())
        roll_v = float(motion_output['roll'].flatten()[0].item())

        full_params = kp_flat + exp_flat + scale_flat + t_flat + [pitch_v, yaw_v, roll_v]
        return full_params

    def predict_inter_frame(self, kp_inter, frame_idx):
        """Decoder: Generate frame from decoded motion parameters.

        kp_inter: dict with 'kp', 'exp', 'scale', 't', 'pitch', 'yaw', 'roll'
        kp_driving = (x_c_s @ R + exp) * scale + t[:2]

        Following the original LivePortrait pipeline, the canonical keypoints
        x_c_s are taken from the source (I-frame) and held fixed for the whole
        sequence. MotionExtractor's per-frame canonical prediction is not
        perfectly identity-invariant, so reusing it per P-frame injects
        high-frequency jitter that manifests as expression flicker / landmark
        drift. Overriding kp_inter['kp'] with the source's canonical removes
        that jitter without changing the transmitted bitstream.
        """
        with torch.no_grad():
            if getattr(self, 'kp_reference_full', None) is not None:
                kp_inter = dict(kp_inter)
                kp_inter['kp'] = self.kp_reference_full['kp']
            kp_driving = self._transform_keypoints(kp_inter)
            warped_features = self.warping_network(
                feature_3d=self.ref_appearance_features,
                kp_driving=kp_driving,
                kp_source=self.kp_reference
            )
            prediction = self.spade_generator(warped_features['out'])
        return prediction

    # -------------------------------------------------------------------------
    # Feature Quantization and Arithmetic Coding
    # -------------------------------------------------------------------------
    def encode_kp(self, kp_list, frame_idx):
        """Quantize parameter list and encode to binary"""
        # 1. Quantize current frame parameters
        quantized_current = [int(round(x * self.q_step[0])) for x in kp_list]

        # 2. Compute residual from reference (frame 0)
        if frame_idx == 0:
            residual = quantized_current
        else:
            previous = self.rec_sem[-1]
            if all(isinstance(x, int) for x in previous):
                previous_quantized = previous
            else:
                previous_quantized = [int(round(x * self.q_step[0])) for x in previous]
            residual = [quantized_current[i] - previous_quantized[i] for i in range(len(quantized_current))]

        # 3. Exp-Golomb encoding
        bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
        final_encoder_expgolomb(residual.copy(), bin_path)

        # Calculate bits
        kp_bits = os.path.getsize(bin_path) * 8
        
        # Store for next residual calculation
        if frame_idx == 0:
            self.rec_sem.append(quantized_current)
        else:
            # Reconstruct and store
            rec_quantized = [previous_quantized[i] + residual[i] for i in range(len(residual))]
            self.rec_sem.append(rec_quantized)

        return kp_list, kp_bits

    def decode_kp(self, frame_idx):
        """Decode from binary, dequantize and reconstruct motion parameters"""
        # 1. Read and decode .bin file
        bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
        
        try:
            kp_dec = final_decoder_expgolomb(bin_path)
            residual_list = data_convert_inverse_expgolomb(kp_dec)
        except FileNotFoundError:
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        
        # 2. Reconstruct quantized parameters
        if frame_idx == 0:
            reconstructed_quantized = residual_list
        else:
            previous = self.rec_sem[-1]
            # I-frame entry in rec_sem may be stored as floats (raw kp_value_frame).
            # Quantize it before adding integer residuals, mirroring encode_kp().
            if all(isinstance(x, int) for x in previous):
                previous_quantized = previous
            else:
                previous_quantized = [int(round(x * self.q_step[0])) for x in previous]
            reconstructed_quantized = [previous_quantized[i] + residual_list[i] for i in range(len(residual_list))]
        
        self.rec_sem.append(reconstructed_quantized)
        
        # 3. Parse parameters
        # Layout: [kp(3*num_kp), exp(3*num_kp), scale(1), t(3), pitch, yaw, roll] = 133 values
        num_kp = self.num_kp
        kp_size = 3 * num_kp
        
        kp_flat = reconstructed_quantized[:kp_size]
        exp_flat = reconstructed_quantized[kp_size:2*kp_size]
        scale_flat = reconstructed_quantized[2*kp_size:2*kp_size+1] if len(reconstructed_quantized) > 2*kp_size else [1.0]
        t_flat = reconstructed_quantized[2*kp_size+1:2*kp_size+4] if len(reconstructed_quantized) > 2*kp_size+1 else [0., 0., 0.]
        # Parse head pose angles (pitch, yaw, roll) - default to 0 if not present
        pose_start = 2*kp_size + 4
        pitch_flat = reconstructed_quantized[pose_start:pose_start+1] if len(reconstructed_quantized) > pose_start else [0.]
        yaw_flat = reconstructed_quantized[pose_start+1:pose_start+2] if len(reconstructed_quantized) > pose_start+1 else [0.]
        roll_flat = reconstructed_quantized[pose_start+2:pose_start+3] if len(reconstructed_quantized) > pose_start+2 else [0.]
        
        # 4. Dequantize and convert to tensors
        kp_tensor = torch.tensor(kp_flat).view(1, num_kp, 3).float().to(self.device) / self.q_step[0]
        exp_tensor = torch.tensor(exp_flat).view(1, num_kp, 3).float().to(self.device) / self.q_step[0]
        scale_tensor = torch.tensor(scale_flat).view(1, 1).float().to(self.device) / self.q_step[0]
        t_tensor = torch.tensor(t_flat).view(1, 3).float().to(self.device) / self.q_step[0]
        pitch_tensor = torch.tensor(pitch_flat).float().to(self.device) / self.q_step[0]
        yaw_tensor = torch.tensor(yaw_flat).float().to(self.device) / self.q_step[0]
        roll_tensor = torch.tensor(roll_flat).float().to(self.device) / self.q_step[0]
        
        dec_motion = {
            'kp': kp_tensor,
            'exp': exp_tensor,
            'scale': scale_tensor,
            't': t_tensor,
            'pitch': pitch_tensor,
            'yaw': yaw_tensor,
            'roll': roll_tensor,
        }
        
        kp_bits = os.path.getsize(bin_path) * 8
        
        return dec_motion, kp_bits

    def decode_kp_closed_loop(self, frame_idx):
        """
        闭环解码模式：解码参数后，用其生成画面，并从画面提取新参数作为参考。
        这与编码端的闭环逻辑对应，确保编解码端的rec_sem一致。
        """
        # 1. 先进行标准解码得到重建参数
        dec_motion, kp_bits = self.decode_kp(frame_idx)
        
        # 2. 【闭环关键】用解码参数生成画面，并从画面提取新参数
        with torch.no_grad():
            kp_driving = self._transform_keypoints(dec_motion)
            warped_features = self.warping_network(
                feature_3d=self.ref_appearance_features,
                kp_driving=kp_driving,
                kp_source=self.kp_reference
            )
            reconstructed_frame = self.spade_generator(warped_features['out'])
            
            # 从重建画面提取运动参数
            new_motion_output = self.motion_extractor(reconstructed_frame)
            new_motion_output = self._normalize_motion_output(new_motion_output)
            
            new_kp_flat = new_motion_output['kp'].flatten().cpu().numpy().tolist()
            new_exp_flat = new_motion_output['exp'].flatten().cpu().numpy().tolist()
            new_scale_flat = new_motion_output['scale'].flatten().cpu().numpy().tolist()
            new_t_flat = new_motion_output['t'].flatten().cpu().numpy().tolist()
            new_pitch_v = float(new_motion_output['pitch'].flatten()[0].item())
            new_yaw_v = float(new_motion_output['yaw'].flatten()[0].item())
            new_roll_v = float(new_motion_output['roll'].flatten()[0].item())
            
            new_params_list = new_kp_flat + new_exp_flat + new_scale_flat + new_t_flat + [new_pitch_v, new_yaw_v, new_roll_v]
            
            # 更新rec_sem为从重建画面提取的参数（替换最后存入的值）
            new_params_quantized = [int(round(x * self.q_step[0])) for x in new_params_list]
            self.rec_sem[-1] = new_params_quantized
        
        return dec_motion, kp_bits

    # -------------------------------------------------------------------------
    # Closed-loop Encoding Support (Experimental)
    # -------------------------------------------------------------------------
    def reconstruct_and_update(self, frame_idx: int) -> torch.Tensor:
        """
        使用前一帧的重建运动参数生成画面，并从中提取新的运动参数作为参考。
        这是闭环预测的核心：编码端模拟解码端的行为。
        
        Returns:
            reconstructed_params: 从重建画面中提取的运动参数列表
        """
        with torch.no_grad():
            # 1. 获取前一帧的重建运动参数（已量化+反量化）
            if len(self.rec_sem) == 0:
                raise RuntimeError("No previous reconstructed parameters available")
            
            prev_reconstructed_quantized = self.rec_sem[-1]
            
            # 2. 将量化值反量化为浮点参数
            prev_params = [x / self.q_step[0] for x in prev_reconstructed_quantized]
            
            # 3. 组装为dec_motion格式
            num_kp = self.num_kp
            kp_size = 3 * num_kp
            
            kp_tensor = torch.tensor(prev_params[:kp_size]).view(1, num_kp, 3).float().to(self.device)
            exp_tensor = torch.tensor(prev_params[kp_size:2*kp_size]).view(1, num_kp, 3).float().to(self.device)
            scale_tensor = torch.tensor(prev_params[2*kp_size:2*kp_size+1]).view(1, 1).float().to(self.device) if len(prev_params) > 2*kp_size else torch.ones(1, 1).to(self.device)
            t_tensor = torch.tensor(prev_params[2*kp_size+1:2*kp_size+4]).view(1, 3).float().to(self.device) if len(prev_params) > 2*kp_size+1 else torch.zeros(1, 3).to(self.device)
            
            pose_start = 2*kp_size + 4
            pitch_tensor = torch.tensor(prev_params[pose_start:pose_start+1]).float().to(self.device) if len(prev_params) > pose_start else torch.zeros(1).to(self.device)
            yaw_tensor = torch.tensor(prev_params[pose_start+1:pose_start+2]).float().to(self.device) if len(prev_params) > pose_start+1 else torch.zeros(1).to(self.device)
            roll_tensor = torch.tensor(prev_params[pose_start+2:pose_start+3]).float().to(self.device) if len(prev_params) > pose_start+2 else torch.zeros(1).to(self.device)
            
            dec_motion = {
                'kp': kp_tensor,
                'exp': exp_tensor,
                'scale': scale_tensor,
                't': t_tensor,
                'pitch': pitch_tensor,
                'yaw': yaw_tensor,
                'roll': roll_tensor,
            }
            
            # 4. 使用当前参考生成画面
            if self.ref_appearance_features is None:
                raise RuntimeError("Reference appearance features not set. Call reference_frame_() first.")
            if self.kp_reference is None:
                raise RuntimeError("Reference keypoints not set. Call kp_reference_() first.")
            
            # 生成画面（模拟解码端）
            kp_driving = self._transform_keypoints(dec_motion)
            warped_features = self.warping_network(
                feature_3d=self.ref_appearance_features,
                kp_driving=kp_driving,
                kp_source=self.kp_reference
            )
            reconstructed_frame = self.spade_generator(warped_features['out'])
            
            # 5. 从重建画面提取运动参数
            new_motion_output = self.motion_extractor(reconstructed_frame)
            new_motion_output = self._normalize_motion_output(new_motion_output)
            
            # 6. 转换为参数列表
            new_kp_flat = new_motion_output['kp'].flatten().cpu().numpy().tolist()
            new_exp_flat = new_motion_output['exp'].flatten().cpu().numpy().tolist()
            new_scale_flat = new_motion_output['scale'].flatten().cpu().numpy().tolist()
            new_t_flat = new_motion_output['t'].flatten().cpu().numpy().tolist()
            new_pitch_v = float(new_motion_output['pitch'].flatten()[0].item())
            new_yaw_v = float(new_motion_output['yaw'].flatten()[0].item())
            new_roll_v = float(new_motion_output['roll'].flatten()[0].item())
            
            new_params_list = new_kp_flat + new_exp_flat + new_scale_flat + new_t_flat + [new_pitch_v, new_yaw_v, new_roll_v]
            
            return new_params_list, reconstructed_frame

    def encode_kp_closed_loop(self, kp_list, frame_idx: int):
        """
        闭环编码模式：使用从重建画面提取的参数作为残差计算的参考
        
        关键区别：
        - 传统GFVC：rec_sem存储原始参数+残差重建值（参数空间闭环）
        - 此模式：rec_sem存储从重建画面提取的新参数（像素空间闭环）
        
        流程：
        1. 从rec_sem获取前一帧从重建画面提取的参考参数
        2. 当前帧原始参数与参考参数计算残差
        3. 编码残差
        4. 重建当前帧参数（从重建画面提取并存储到rec_sem供下一帧使用）
        """
        # 量化当前帧原始参数
        quantized_current = [int(round(x * self.q_step[0])) for x in kp_list]
        
        if frame_idx == 0:
            # I帧：编码原始参数，但从重建画面提取参数存入rec_sem供下一帧参考
            # Step 1: 编码I帧原始参数（残差为0）
            bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
            final_encoder_expgolomb(quantized_current.copy(), bin_path)
            kp_bits = os.path.getsize(bin_path) * 8
            
            # Step 2: 【闭环关键】用I帧参数生成画面，并从画面提取新参数
            rec_params_float = [x / self.q_step[0] for x in quantized_current]
            num_kp = self.num_kp
            kp_size = 3 * num_kp
            dec_motion = {
                'kp': torch.tensor(rec_params_float[:kp_size]).view(1, num_kp, 3).float().to(self.device),
                'exp': torch.tensor(rec_params_float[kp_size:2*kp_size]).view(1, num_kp, 3).float().to(self.device),
                'scale': torch.tensor(rec_params_float[2*kp_size:2*kp_size+1]).view(1, 1).float().to(self.device) if len(rec_params_float) > 2*kp_size else torch.ones(1, 1).to(self.device),
                't': torch.tensor(rec_params_float[2*kp_size+1:2*kp_size+4]).view(1, 3).float().to(self.device) if len(rec_params_float) > 2*kp_size+1 else torch.zeros(1, 3).to(self.device),
                'pitch': torch.tensor(rec_params_float[2*kp_size+4:2*kp_size+5]).float().to(self.device) if len(rec_params_float) > 2*kp_size+4 else torch.zeros(1).to(self.device),
                'yaw': torch.tensor(rec_params_float[2*kp_size+5:2*kp_size+6]).float().to(self.device) if len(rec_params_float) > 2*kp_size+5 else torch.zeros(1).to(self.device),
                'roll': torch.tensor(rec_params_float[2*kp_size+6:2*kp_size+7]).float().to(self.device) if len(rec_params_float) > 2*kp_size+6 else torch.zeros(1).to(self.device),
            }
            
            with torch.no_grad():
                kp_driving = self._transform_keypoints(dec_motion)
                warped_features = self.warping_network(
                    feature_3d=self.ref_appearance_features,
                    kp_driving=kp_driving,
                    kp_source=self.kp_reference
                )
                reconstructed_frame = self.spade_generator(warped_features['out'])
                
                # 从重建画面提取运动参数
                new_motion_output = self.motion_extractor(reconstructed_frame)
                new_motion_output = self._normalize_motion_output(new_motion_output)
                
                new_kp_flat = new_motion_output['kp'].flatten().cpu().numpy().tolist()
                new_exp_flat = new_motion_output['exp'].flatten().cpu().numpy().tolist()
                new_scale_flat = new_motion_output['scale'].flatten().cpu().numpy().tolist()
                new_t_flat = new_motion_output['t'].flatten().cpu().numpy().tolist()
                new_pitch_v = float(new_motion_output['pitch'].flatten()[0].item())
                new_yaw_v = float(new_motion_output['yaw'].flatten()[0].item())
                new_roll_v = float(new_motion_output['roll'].flatten()[0].item())
                
                new_params_list = new_kp_flat + new_exp_flat + new_scale_flat + new_t_flat + [new_pitch_v, new_yaw_v, new_roll_v]
                new_params_quantized = [int(round(x * self.q_step[0])) for x in new_params_list]
                
                # 【关键】将从重建画面提取的参数量化并存入rec_sem
                self.rec_sem.append(new_params_quantized)
            
            rec_params = new_params_list
        else:
            # P帧：闭环预测
            # Step 1: 获取前一帧从重建画面提取的参数（存储在rec_sem中）
            previous_reconstructed_from_frame = self.rec_sem[-1]
            
            # Step 2: 计算残差：当前原始参数 - 前一帧重建画面提取参数
            residual = [int(quantized_current[i] - previous_reconstructed_from_frame[i])
                       for i in range(len(quantized_current))]
            
            # Step 3: 编码残差
            bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
            final_encoder_expgolomb(residual.copy(), bin_path)
            kp_bits = os.path.getsize(bin_path) * 8
            
            # Step 4: 重建当前帧参数（在参数空间）
            rec_quantized = [previous_reconstructed_from_frame[i] + residual[i] 
                            for i in range(len(residual))]
            
            # Step 5: 【闭环关键】用重建参数生成画面，并从画面提取新参数
            # 反量化为浮点
            rec_params_float = [x / self.q_step[0] for x in rec_quantized]
            
            # 组装为dec_motion格式并生成画面
            num_kp = self.num_kp
            kp_size = 3 * num_kp
            dec_motion = {
                'kp': torch.tensor(rec_params_float[:kp_size]).view(1, num_kp, 3).float().to(self.device),
                'exp': torch.tensor(rec_params_float[kp_size:2*kp_size]).view(1, num_kp, 3).float().to(self.device),
                'scale': torch.tensor(rec_params_float[2*kp_size:2*kp_size+1]).view(1, 1).float().to(self.device) if len(rec_params_float) > 2*kp_size else torch.ones(1, 1).to(self.device),
                't': torch.tensor(rec_params_float[2*kp_size+1:2*kp_size+4]).view(1, 3).float().to(self.device) if len(rec_params_float) > 2*kp_size+1 else torch.zeros(1, 3).to(self.device),
                'pitch': torch.tensor(rec_params_float[2*kp_size+4:2*kp_size+5]).float().to(self.device) if len(rec_params_float) > 2*kp_size+4 else torch.zeros(1).to(self.device),
                'yaw': torch.tensor(rec_params_float[2*kp_size+5:2*kp_size+6]).float().to(self.device) if len(rec_params_float) > 2*kp_size+5 else torch.zeros(1).to(self.device),
                'roll': torch.tensor(rec_params_float[2*kp_size+6:2*kp_size+7]).float().to(self.device) if len(rec_params_float) > 2*kp_size+6 else torch.zeros(1).to(self.device),
            }
            
            with torch.no_grad():
                # 生成重建画面
                kp_driving = self._transform_keypoints(dec_motion)
                warped_features = self.warping_network(
                    feature_3d=self.ref_appearance_features,
                    kp_driving=kp_driving,
                    kp_source=self.kp_reference
                )
                reconstructed_frame = self.spade_generator(warped_features['out'])
                
                # 【关键】从重建画面重新提取运动参数
                new_motion_output = self.motion_extractor(reconstructed_frame)
                new_motion_output = self._normalize_motion_output(new_motion_output)
                
                # 转换为列表并量化
                new_kp_flat = new_motion_output['kp'].flatten().cpu().numpy().tolist()
                new_exp_flat = new_motion_output['exp'].flatten().cpu().numpy().tolist()
                new_scale_flat = new_motion_output['scale'].flatten().cpu().numpy().tolist()
                new_t_flat = new_motion_output['t'].flatten().cpu().numpy().tolist()
                new_pitch_v = float(new_motion_output['pitch'].flatten()[0].item())
                new_yaw_v = float(new_motion_output['yaw'].flatten()[0].item())
                new_roll_v = float(new_motion_output['roll'].flatten()[0].item())
                
                new_params_list = new_kp_flat + new_exp_flat + new_scale_flat + new_t_flat + [new_pitch_v, new_yaw_v, new_roll_v]
                
                # 【优化】直接从重建画面提取的参数量化，减少二次量化误差
                # 注意：残差计算需要量化值，所以这里仍然要量化
                # 但使用更精确的舍入方式
                new_params_quantized = [int(round(x * self.q_step[0])) for x in new_params_list]
                self.rec_sem.append(new_params_quantized)
            
            rec_params = new_params_list  # 返回从重建画面提取的参数（浮点）
        
        return rec_params, kp_bits
