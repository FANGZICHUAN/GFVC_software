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

# new added start
import compressai
from compressai.zoo import models
# new added end
from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
#from compressai.layers import GDN, MaskedConv2d
#from compressai.models.utils import conv, deconv, update_registered_buffers

import struct
# import torchac

# from modules.selfcompressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv

# -

class VideoCompressor(nn.Module):
    #def __init__(self, out_channel_mv,IsTraining):
    def __init__(self, out_channel_mv):

        super(VideoCompressor, self).__init__()
        
        self.out_channel_mv = out_channel_mv
        
        #self.istraining = IsTraining
        
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
        
    # From Balle's tensorflow compression examples
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64


    def get_scale_table(self, min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
    
    
    
    def forward(self,  Driving_Feature, Source_Feature):
        
        self.entropy_bottleneck.eval()
        self.entropy_bottleneck.update() #force=True

        self.gaussian_conditional.eval()
        # self.gaussian_conditional.update() #force=True

        scale_table = self.get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=False) #False
        

        y = Driving_Feature['residual'] - Source_Feature['residual']

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)            
        
        return y_strings, z_strings
    
    
    

    