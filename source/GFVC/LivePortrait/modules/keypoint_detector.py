"""
LivePortrait Keypoint Detector
Detects 3D implicit keypoints for face animation
Based on face-vid2vid approach with enhanced 3D modeling
"""
from torch import nn
import torch
from torchvision import models


class KPDetector(nn.Module):
    """
    3D Implicit Keypoint Detector for LivePortrait
    Predicts K 3D keypoints (x, y, z) for head pose and facial expression
    """

    def __init__(self, num_kp=64, **kwargs):
        super(KPDetector, self).__init__()
        self.num_kp = num_kp

        # Use ResNet-18 as backbone for feature extraction
        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        # Output: num_kp * 3 (x, y, z coordinates for each keypoint)
        self.fg_encoder.fc = nn.Linear(num_features, num_kp * 3)

    def forward(self, image):
        # Extract 3D keypoints
        fg_kp = self.fg_encoder(image)
        bs = fg_kp.shape[0]
        
        # Normalize to [-1, 1] range
        fg_kp = torch.sigmoid(fg_kp)
        fg_kp = fg_kp * 2 - 1
        
        # Reshape to (batch, num_kp, 3) for 3D coordinates
        out = {'value': fg_kp.view(bs, self.num_kp, 3)}
        return out
