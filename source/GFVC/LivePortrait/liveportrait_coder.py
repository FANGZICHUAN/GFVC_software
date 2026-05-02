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
        """Store I-frame as reference and extract appearance features"""
        self.reference_frame = reference
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
        kp_driving = (kp @ R + exp) * scale + t[:2]
        """
        with torch.no_grad():
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
            residual = [quantized_current[i] - self.rec_sem[-1][i] for i in range(len(quantized_current))]

        # 3. Exp-Golomb encoding
        bin_path = os.path.join(self.kp_output_dir, 'frame' + str(frame_idx).zfill(4) + '.bin')
        final_encoder_expgolomb(residual, bin_path)

        # Calculate bits
        kp_bits = os.path.getsize(bin_path) * 8
        
        # Store for next residual calculation
        if frame_idx == 0:
            self.rec_sem.append(quantized_current)
        else:
            # Reconstruct and store
            rec_quantized = [self.rec_sem[-1][i] + residual[i] for i in range(len(residual))]
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
            reconstructed_quantized = [self.rec_sem[-1][i] + residual_list[i] for i in range(len(residual_list))]
        
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
