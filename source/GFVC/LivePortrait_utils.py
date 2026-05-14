import os
import yaml
import numpy as np
import torch
from typing import Dict, List
import json

from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from GFVC.LivePortrait.modules.motion_extractor import MotionExtractor
from GFVC.LivePortrait.modules.warping_network import WarpingNetwork
from GFVC.LivePortrait.modules.appearance_feature_extractor import AppearanceFeatureExtractor
from GFVC.LivePortrait.modules.spade_generator import SPADEDecoder


def load_liveportrait_checkpoints(config_path, checkpoint_dir, device='cpu'):
    """Load LivePortrait model checkpoints from separate .pth files
    
    Args:
        config_path: Path to model config YAML file
        checkpoint_dir: Directory containing the 4 checkpoint files
        device: Device to load models on
    
    Expected checkpoint files in checkpoint_dir:
        - appearance_feature_extractor.pth
        - motion_extractor.pth
        - warping_module.pth
        - spade_generator.pth
    """
    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Initialize networks
    # AppearanceFeatureExtractor - explicit parameters
    appearance_params = config['model_params'].get('appearance_params', {})
    appearance_extractor = AppearanceFeatureExtractor(
        image_channel=appearance_params.get('image_channel', 3),
        block_expansion=appearance_params.get('block_expansion', 64),
        num_down_blocks=appearance_params.get('num_down_blocks', 4),
        max_features=appearance_params.get('max_features', 512),
        reshape_channel=appearance_params.get('reshape_channel', 32),
        reshape_depth=appearance_params.get('reshape_depth', 16),
        num_resblocks=appearance_params.get('num_resblocks', 2)
    )
    appearance_extractor.to(device)

    motion_extractor = MotionExtractor(**config['model_params']['common_params'])
    motion_extractor.to(device)

    # WarpingNetwork - explicit parameters
    warp_params = config['model_params']['warping_params']
    warping_network = WarpingNetwork(
        num_kp=warp_params.get('num_kp', 21),
        block_expansion=warp_params.get('block_expansion', 64),
        max_features=warp_params.get('max_features', 512),
        num_down_blocks=warp_params.get('num_down_blocks', 2),
        reshape_channel=warp_params.get('reshape_channel', 32),
        estimate_occlusion_map=warp_params.get('estimate_occlusion_map', True),
        dense_motion_params=warp_params.get('dense_motion_params', {
            'block_expansion': 32, 'num_blocks': 5, 'max_features': 1024, 'reshape_depth': 16, 'compress': 4
        })
    )
    warping_network.to(device)

    # SPADEDecoder - explicit parameters (no num_kp)
    gen_params = config['model_params'].get('generator_params', {})
    spade_generator = SPADEDecoder(
        upscale=gen_params.get('upscale', 2),
        max_features=gen_params.get('max_features', 512),
        block_expansion=gen_params.get('block_expansion', 64),
        out_channels=gen_params.get('out_channels', 64),
        num_down_blocks=gen_params.get('num_down_blocks', 2)
    )
    spade_generator.to(device)

    # Load individual checkpoint files
    appearance_extractor.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'appearance_feature_extractor.pth'), 
                  map_location=device), strict=True
    )
    motion_extractor.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'motion_extractor.pth'), 
                  map_location=device), strict=True
    )
    warping_network.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'warping_module.pth'), 
                  map_location=device), strict=True
    )
    spade_generator.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'spade_generator.pth'), 
                  map_location=device), strict=True
    )

    # Set eval mode
    appearance_extractor.eval()
    motion_extractor.eval()
    warping_network.eval()
    spade_generator.eval()

    return appearance_extractor, motion_extractor, warping_network, spade_generator


class LivePortrait:
    '''LivePortrait Encoder/Decoder'''
    def __init__(self, model_config_path: str = None, model_checkpoint_path: str = None, 
                 kp_output_dir: str = None, q_step: List[int] = [128, 128], device: str = 'cpu') -> None:
        self.device = device
        if model_checkpoint_path is not None:
            # model_checkpoint_path should be the checkpoint directory
            appearance_extractor, motion_extractor, warping_network, spade_generator = \
                load_liveportrait_checkpoints(model_config_path, model_checkpoint_path, device=device)
            self.appearance_extractor = appearance_extractor.to(device)
            self.analysis_model = motion_extractor.to(device)  # motion_extractor = keypoint detector
            self.warping_network = warping_network.to(device)
            self.synthesis_model = spade_generator.to(device)
            self.reference_frame = None
            self.kp_reference = None
            self.ref_appearance_features = None
        if q_step is not None:
            self.kp_output_dir = kp_output_dir
            self.q_step = q_step
            self.rec_sem = []

    def reference_frame_(self, x):
        self.reference_frame = x
        # Extract appearance features from reference frame
        with torch.no_grad():
            self.ref_appearance_features = self.appearance_extractor(x)

    def kp_reference_(self, motion_output: Dict[str, torch.Tensor]):
        """Store reference keypoints from MotionExtractor output
        
        Reference keypoints = kp_canonical (without expression)
        """
        # Use canonical keypoints as reference (without expression deformation)
        self.kp_reference = motion_output['kp']  # (B, num_kp, 3)

    def get_kp_list(self, motion_output: Dict[str, torch.Tensor], frame_idx: int, save: bool = False) -> List[str]:
        """Extract keypoint list from MotionExtractor output
        
        MotionExtractor returns:
        - kp: canonical keypoints (B, 3*num_kp)
        - exp: expression deformation (B, 3*num_kp)
        - pitch, yaw, roll: head pose
        - t: translation (B, 3)
        - scale: scale factor (B, 1)
        
        For compression, we encode kp + exp + pose parameters
        """
        # Extract components
        kp = motion_output['kp']  # (B, 3*num_kp)
        exp = motion_output['exp']  # (B, 3*num_kp)
        
        # Combine canonical keypoints with expression deformation
        # Final keypoints = kp + exp
        combined = kp + exp
        
        # Flatten to list
        combined_list = combined.flatten().tolist()
        
        # Also include scale and translation for pose info (optional, for better quality)
        scale = motion_output['scale'].flatten().tolist()
        t = motion_output['t'].flatten().tolist()
        
        # Concatenate all parameters
        full_params = combined_list + scale + t
        
        full_params_str = str(full_params)
        full_params_str = "".join(full_params_str.split())

        if save:
            with open(self.kp_output_dir + '/frame' + str(frame_idx) + '.txt', 'w') as f:
                f.write(full_params_str)

        kp_value_frame = json.loads(full_params_str)
        kp_value_frame = eval('[%s]' % repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame

    def encode_kp(self, kp_list: List[str], frame_idx: int):
        """Encode keypoints with residual coding"""
        # Compute residual
        if len(self.rec_sem) == 0:
            kp_difference = (np.array(kp_list)).tolist()
        else:
            kp_difference = (np.array(kp_list) - np.array(self.rec_sem[-1])).tolist()

        # Quantization (3D keypoints need 3 channels)
        num_kp = len(kp_difference) // 3
        kp_x = kp_difference[:num_kp]
        kp_y = kp_difference[num_kp:2*num_kp]
        kp_z = kp_difference[2*num_kp:]

        kp_x = [int(i * self.q_step[0]) for i in kp_x]
        kp_y = [int(i * self.q_step[0]) for i in kp_y]
        kp_z = [int(i * self.q_step[1]) for i in kp_z]

        kp_difference = kp_x + kp_y + kp_z
        kp_difference = list(map(round, kp_difference))

        bin_file = self.kp_output_dir + '/frame' + str(frame_idx).zfill(4) + '.bin'
        final_encoder_expgolomb(kp_difference, bin_file)

        bits = os.path.getsize(bin_file) * 8

        # Decoding for residual reconstruction
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)

        # Inverse quantization
        num_res = len(res_difference_dec) // 3
        res_x = res_difference_dec[:num_res]
        res_y = res_difference_dec[num_res:2*num_res]
        res_z = res_difference_dec[2*num_res:]

        res_x = [i / self.q_step[0] for i in res_x]
        res_y = [i / self.q_step[0] for i in res_y]
        res_z = [i / self.q_step[1] for i in res_z]

        res_difference_dec = res_x + res_y + res_z

        if len(self.rec_sem) == 0:
            rec_semantics = (np.array(res_difference_dec)).tolist()
        else:
            rec_semantics = (np.array(res_difference_dec) + np.array(self.rec_sem[-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits

    def decode_kp(self, frame_idx: int):
        """Decode keypoints from binary file
        
        Returns MotionExtractor-compatible output with:
        - kp: canonical keypoints
        - exp: expression deformation  
        - scale: scale factor
        - t: translation
        """
        frame_index = str(frame_idx).zfill(4)
        bin_file = self.kp_output_dir + '/frame' + frame_index + '.bin'
        bits = os.path.getsize(bin_file) * 8
        kp_dec = final_decoder_expgolomb(bin_file)

        # Decode residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)

        # Inverse quantization (all params use same q_step[0])
        kp_difference_dec = [i / self.q_step[0] for i in kp_difference]
        kp_difference_dec = eval('[%s]' % repr(kp_difference_dec).replace('[', '').replace(']', ''))

        if len(self.rec_sem) == 0:
            kp_previous = [0. for _ in range(len(kp_difference_dec))]
        else:
            kp_previous = eval('[%s]' % repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))

        # Reconstruct parameters
        kp_integer = [kp_previous[i] + kp_difference_dec[i] for i in range(len(kp_difference_dec))]
        self.rec_sem.append(kp_integer)

        # Parse the reconstructed parameters
        # Layout: [kp(3*64), exp(3*64), scale(1), t(3)]
        num_kp = 64  # Default num_kp for LivePortrait
        kp_size = 3 * num_kp
        
        kp_flat = kp_integer[:kp_size]
        exp_flat = kp_integer[kp_size:2*kp_size]
        scale_flat = kp_integer[2*kp_size:2*kp_size+1] if len(kp_integer) > 2*kp_size else [1.0]
        t_flat = kp_integer[2*kp_size+1:2*kp_size+4] if len(kp_integer) > 2*kp_size+1 else [0., 0., 0.]

        # Convert to tensors
        kp_tensor = torch.Tensor(kp_flat).reshape((1, num_kp, 3)).to(self.device)
        exp_tensor = torch.Tensor(exp_flat).reshape((1, num_kp, 3)).to(self.device)
        scale_tensor = torch.Tensor(scale_flat).reshape((1, 1)).to(self.device)
        t_tensor = torch.Tensor(t_flat).reshape((1, 3)).to(self.device)

        # Return MotionExtractor-compatible dict
        kp_inter_frame = {
            'kp': kp_tensor,
            'exp': exp_tensor,
            'scale': scale_tensor,
            't': t_tensor,
        }

        return kp_inter_frame, bits

    def predict_inter_frame(self, kp_inter_frame: Dict[str, torch.Tensor], frame_idx: int) -> torch.Tensor:
        """Generate frame from keypoints using warping + SPADE generator
        
        Args:
            kp_inter_frame: MotionExtractor output dict with 'kp', 'exp', 'scale', 't'
        """
        # Combine canonical keypoints with expression deformation
        # Driving keypoints = kp_canonical + exp
        kp_driving = kp_inter_frame['kp'] + kp_inter_frame['exp']  # (B, num_kp, 3)
        
        # Apply scale and translation to driving keypoints
        scale = kp_inter_frame.get('scale', torch.ones(1, 1).to(self.device))
        t = kp_inter_frame.get('t', torch.zeros(1, 3).to(self.device))
        
        # Scale and translate
        kp_driving = kp_driving * scale.unsqueeze(1) + t.unsqueeze(1)
        
        # Warping stage - use appearance features
        warped_features = self.warping_network(
            self.ref_appearance_features, 
            kp_driving, 
            self.kp_reference
        )
        
        # SPADE generator stage - takes warped feature 'out'
        prediction = self.synthesis_model(warped_features['out'])
        return prediction
