import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F
from GFVC.SRLR.modules.util import Hourglass, AntiAliasInterpolation2d
from GFVC.SRLR.modules.GDN import GDN
import math


    
    
class Residual_Detector(nn.Module):
    # def __init__(self, block_expansion, num_kp, num_channels, num_ref, max_features,
    #              num_blocks, temperature, codec_mode,generator_mode, estimate_jacobian=False, scale_factor=1,
    #              single_jacobian_map=False, pad=0):
    def __init__(self, block_expansion, num_kp, num_channels, num_ref, max_features,
                 num_blocks, temperature, codec_mode, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):        
        super(Residual_Detector, self).__init__()
        """
        source code from  https://github.com/balajiselvaraj1601/RCF_Pytorch_Updated
        """
        
        self.scale_factor = scale_factor
        # if self.scale_factor != 1:
        #     self.down = AntiAliasInterpolation2d(3, self.scale_factor)
            
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
           
                
        ##feature map/heatmap—> latent code (Analysis net)
        self.conv1 = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + num_kp) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(num_kp)
        
        self.conv2 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(num_kp)
                 
        self.conv3 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(num_kp)
        
        self.conv4 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.gdn4 = GDN(num_kp)        
            
        self.conv5 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv5.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv5.bias.data, 0.01)
        self.gdn5 = GDN(num_kp)              
            
        self.conv6 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv6.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv6.bias.data, 0.01)
        self.gdn6 = GDN(num_kp)
        
        self.conv7 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv7.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv7.bias.data, 0.01)
        self.gdn7 = GDN(num_kp)        
        
        self.conv8 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv8.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv8.bias.data, 0.01)
        self.gdn8 = GDN(num_kp)        
        
        self.conv9 = nn.Conv2d(in_channels=num_kp,out_channels=num_kp, kernel_size=(5, 5), stride=1, padding=0)
        torch.nn.init.xavier_normal_(self.conv9.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv9.bias.data, 0.01)
        self.gdn9 = GDN(num_kp)  
            
        
    def forward(self, x):

        
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(3, 0.5).cuda()        
            x = self.down(x)    #torch.Size([bs, 3, 128, 128]) 
        feature_map = self.predictor(x)  #torch.Size([bs, 35, 128, 128])
        
        if self.scale_factor ==1/4:  
            predict_residual=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            # print(predict_residual.shape)   

        if self.scale_factor ==3/16:
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            prediction_gdn2=self.gdn6(self.conv6(prediction_gdn1))  #torch.Size([bs, 1, 24, 24])    
            prediction_gdn3=self.gdn7(self.conv7(prediction_gdn2))  #torch.Size([bs, 1, 24, 24])   
            prediction_gdn4=self.gdn8(self.conv8(prediction_gdn3))  #torch.Size([bs, 1, 24, 24])   
            predict_residual=self.gdn9(self.conv9(prediction_gdn4))  #torch.Size([bs, 1, 24, 24])   
            
            
        if self.scale_factor ==1/8:
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            predict_residual=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, 1, 32, 32])         
            
        if self.scale_factor ==3/32:
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            prediction_gdn2=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, 1, 32, 32])   
            prediction_gdn3=self.gdn6(self.conv6(prediction_gdn2))  #torch.Size([bs, 1, 24, 24])    
            predict_residual=self.gdn7(self.conv7(prediction_gdn3))  #torch.Size([bs, 1, 24, 24])    
            
            
            
        if self.scale_factor ==1/16:
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            prediction_gdn2=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, 1, 32, 32])       
            predict_residual=self.gdn3(self.conv3(prediction_gdn2))  #torch.Size([bs, 1, 16, 16])    
            
        if self.scale_factor ==1/32:
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            prediction_gdn2=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, 1, 32, 32])       
            prediction_gdn3=self.gdn3(self.conv3(prediction_gdn2))  #torch.Size([bs, 1, 16, 16])     
            predict_residual=self.gdn4(self.conv4(prediction_gdn3))  #torch.Size([bs, 1, 8, 8])                                  
            
        if self.scale_factor ==1/64:        
            prediction_gdn1=self.gdn1(self.conv1(feature_map))  #torch.Size([bs, 1, 64, 64])
            prediction_gdn2=self.gdn2(self.conv2(prediction_gdn1))  #torch.Size([bs, 1, 32, 32])       
            prediction_gdn3=self.gdn3(self.conv3(prediction_gdn2))  #torch.Size([bs, 1, 16, 16])     
            prediction_gdn4=self.gdn4(self.conv4(prediction_gdn3))  #torch.Size([bs, 1, 8, 8])    
            predict_residual=self.gdn5(self.conv5(prediction_gdn4))  #torch.Size([bs, 1, 4, 4]) 
            
        # print(predict_residual.shape)

        out = {'residual': predict_residual}  

        return out            
        

    
# Based on BDCN Implementation @ https://github.com/pkuCactus/BDCN
def crop(data1, h, w , crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert(h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h+h, crop_w:crop_w+w]
    return data

def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)