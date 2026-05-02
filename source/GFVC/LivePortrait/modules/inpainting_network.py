"""
LivePortrait Inpainting Network
Generates final output frame from warped features
Enhanced with regional GAN losses for better facial details
"""
from torch import nn
import torch
import torch.nn.functional as F


class InpaintingNetwork(nn.Module):
    """
    Generator network for face reconstruction
    Uses U-Net architecture with skip connections
    """

    def __init__(self, num_channels=3, block_expansion=64, num_blocks=5, 
                 max_features=512, **kwargs):
        super(InpaintingNetwork, self).__init__()
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = num_channels
        for i in range(num_blocks):
            out_ch = min(block_expansion * (2**i), max_features)
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
            in_ch = out_ch
        
        # Decoder with skip connections
        self.decoder = nn.ModuleList()
        for i in range(num_blocks-1, -1, -1):
            in_ch = min(block_expansion * (2**i), max_features)
            out_ch = min(block_expansion * (2**(i-1)), max_features) if i > 0 else num_channels
            
            self.decoder.append(
                nn.Sequential(
                    nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),  # *2 for skip
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, warped_features):
        """
        Args:
            warped_features: Warped feature tensor from warping network
        Returns:
            prediction: Final generated frame
        """
        # Encode
        enc_outputs = []
        x = warped_features
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        # Decode with skip connections
        for i, layer in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            # Skip connection
            skip = enc_outputs[-(i+1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = layer(x)
        
        # Final output
        prediction = self.final(x)
        
        return {'prediction': prediction}
