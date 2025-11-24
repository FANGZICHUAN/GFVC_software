# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter

# # new added start
# import compressai
# from compressai.zoo import models
# # new added end
# from compressai.models import CompressionModel
# from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
# #from compressai.layers import GDN, MaskedConv2d
# #from compressai.models.utils import conv, deconv, update_registered_buffers
from compressai.models.utils import conv, deconv
import struct
# import torchac

#from modules.selfcompressai.entropy_models import EntropyBottleneck, GaussianConditional
# -



class VideoCompressor(nn.Module):
    
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    .. code-block:: none

                  ┌───┐    y     ┌───┐  z  ┌───┐ z_hat      z_hat ┌───┐
            x ──►─┤g_a├──►─┬──►──┤h_a├──►──┤ Q ├───►───·⋯⋯·───►───┤h_s├─┐
                  └───┘    │     └───┘     └───┘        EB        └───┘ │
                           ▼                                            │
                         ┌─┴─┐                                          │
                         │ Q │                                          ▼
                         └─┬─┘                                          │
                           │                                            │
                     y_hat ▼                                            │
                           │                                            │
                           ·                                            │
                        GC : ◄─────────────────────◄────────────────────┘
                           ·                 scales_hat
                           │                 means_hat
                     y_hat ▼
                           │
                  ┌───┐    │
        x_hat ──◄─┤g_s├────┘
                  └───┘

        EB = Entropy bottleneck
        GC = Gaussian conditional

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """ 
    
    
    
    def __init__(self, out_channel_mv):
        super(VideoCompressor, self).__init__()
        
        self.out_channel_mv = out_channel_mv
                
        self.entropy_bottleneck = EntropyBottleneck(self.out_channel_mv)
        

        N=1
        M=1
    
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )    
    

        self.gaussian_conditional = GaussianConditional(None)
    
        

    def forward(self,  Driving_Feature, Source_Feature):
        
        fs = Driving_Feature['residual'] - Source_Feature['residual']
        # print(fs.shape)                
        bs, c, h , w= fs.shape  ## bs,1,4,4
                
        z = self.h_a(fs)
        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        
        fs_hat, fs_likelihoods = self.gaussian_conditional(fs, scales_hat, means=means_hat)
        
        total_bits = torch.log(fs_likelihoods).sum() / (-math.log(2))   + torch.log(z_likelihoods).sum() / (-math.log(2))   

        quant_driving = fs_hat + Source_Feature['residual']
        quant_driving = {'residual': quant_driving}  ####

        return total_bits, quant_driving       
    
