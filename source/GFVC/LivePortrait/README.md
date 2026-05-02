# LivePortrait Integration for GFVC Software

This directory contains the LivePortrait model integration for the GFVC (Generative Face Video Coding) software.

## Overview

LivePortrait is a state-of-the-art face animation model that uses 3D implicit keypoints for high-quality talking head generation. This implementation adapts LivePortrait for video compression tasks.

## Key Features

- **3D Implicit Keypoints**: Uses 64 3D keypoints (x, y, z) for robust head pose and facial expression modeling
- **High Fidelity**: Enhanced perceptual quality with regional GAN losses
- **Low Bitrate**: Achieves perceptually lossless compression at ~22kbps
- **Real-time Capability**: Optimized for video conferencing applications

## Architecture

### Core Components

1. **Keypoint Detector** (`modules/keypoint_detector.py`)
   - ResNet-18 backbone
   - Outputs 64 3D keypoints per frame
   - Normalized to [-1, 1] range

2. **Warping Network** (`modules/warping_network.py`)
   - Feature pyramid encoder
   - Keypoint-based attention mechanism
   - Optical flow generation
   - Occlusion mask prediction

3. **Inpainting Network** (`modules/inpainting_network.py`)
   - U-Net architecture with skip connections
   - Generates final output from warped features
   - Enhanced facial detail reconstruction

### Compression Pipeline

```
Input Frame → Keypoint Detector → 3D Keypoints → Quantization → Arithmetic Coding
                                                        ↓
Reference Frame ← Inpainting ← Warping ← Decoded Keypoints
```

## Usage

### Configuration

Edit the config file at `config_256/LivePortrait_VOXCELEB_enc.yaml`:

```yaml
codec_name: 'LivePortrait'
dataset: 'VOXCELEB'
height: 256
width: 256
iframe_format: 'YUV420'
qp_list: ["12","22","32","42"]
feature_coding_mode: 'SEI'
ae_qf: [128,128]  # Quantization factors for 3D keypoints
```

### Running Experiments

1. **Single Model Run**:
   ```bash
   cd source
   python run.py --config ../config_256/LivePortrait_VOXCELEB_enc.yaml
   ```

2. **Batch Evaluation with run_all.py**:
   The LivePortrait model is automatically included in `run_all.py` MODELS list.

### Checkpoint Setup

Place the pretrained LivePortrait checkpoint at:
```
source/GFVC/LivePortrait/checkpoint/LivePortrait-checkpoint.pth.tar
```

The checkpoint should contain:
- `kp_detector`: Keypoint detector weights
- `warping_network`: Warping network weights  
- `inpainting_network`: Inpainting network weights

## Implementation Notes

### Differences from TPSMM

1. **3D vs 2D Keypoints**: LivePortrait uses 3D (x, y, z) keypoints while TPSMM uses 2D (x, y)
2. **Network Architecture**: LivePortrait uses separate warping and inpainting networks
3. **Quantization**: Requires 3-channel quantization for z-coordinate
4. **Feature Range**: Keypoints normalized to [-1, 1] with 128x quantization factor

### Integration with GFVC Software

- **Encoder**: `liveportrait_coder.py` provides `LivePortraitCoder` class
- **Utils**: `LivePortrait_utils.py` provides model loading and codec interface
- **Codec Registration**: Added to `codecs_utils.py::get_inter_coder()`
- **Run All**: Added to `run_all.py::MODELS` list

## References

Based on:
- LivePortrait: Efficient Portrait Animation with Stitching and Retargeting Control (2024)
- face-vid2vid: One-Shot Free-View Neural Talking-Head Synthesis (CVPR 2021)

## TODO

- [ ] Download and place official LivePortrait checkpoint
- [ ] Verify 3D keypoint quantization parameters
- [ ] Test on VOXCELEB dataset
- [ ] Compare RD performance with CFTE/TPSMM
- [ ] Implement feature translation with other GFVC models
