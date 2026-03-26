import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn as nn 

from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path


from GFVC.CFTE.sync_batchnorm import DataParallelWithCallback
from GFVC.FV2V_utils import headpose_pred_to_degree, get_rotation_matrix

def load_CFTE_Enc_translator(checkpoint_path, trans_channel, cpu=False):

    translator = CFTE_Enc(N=trans_channel)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  

    translator.load_state_dict(checkpoint['state_dict']['CFTE_Enc'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_CFTE_Dec_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = CFTE_Dec(N=trans_channel)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')   
    translator.load_state_dict(checkpoint['state_dict']['CFTE_Dec'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_FOMM_Enc_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = FOMM_Enc(N=trans_channel)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')   
    translator.load_state_dict(checkpoint['state_dict']['FOMM_Enc'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator


def load_FOMM_Dec_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = FOMM_Dec(N=trans_channel)
    
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    translator.load_state_dict(checkpoint['state_dict']['FOMM_Dec'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_FV2V_Enc_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = FV2V_Enc(N=trans_channel)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    translator.load_state_dict(checkpoint['state_dict']['FV2V_Enc'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_FV2V_Dec_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = FV2V_Dec(N=trans_channel)

    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    translator.load_state_dict(checkpoint['state_dict']['FV2V_Dec'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_DAC_Enc_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = DAC_Enc(N=trans_channel)

    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    translator.load_state_dict(checkpoint['state_dict']['DAC_Enc'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator

def load_DAC_Dec_translator(checkpoint_path, trans_channel,cpu=False):
    
    translator = DAC_Dec(N=trans_channel)

    checkpoint = torch.load(checkpoint_path, map_location='cpu') 
    translator.load_state_dict(checkpoint['state_dict']['DAC_Dec'], strict=True)
        
        
    if not cpu:
        translator.cuda()
        translator = DataParallelWithCallback(translator)

    translator.eval()
    return translator



class CFTE_Enc(nn.Module):
    def __init__(self, N=64):
        super(CFTE_Enc, self).__init__()
    
        self.CFTE = nn.Sequential(
            nn.Linear(16,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            
        )
    

    def forward(self,x,x_ref=None,use_delta=False):
        x = x['value']
        if x_ref is not None and use_delta:
            x_ref = x_ref['value']
            x = x-x_ref
        mid = self.CFTE(x.reshape(x.size(0),16))
        
        return mid

class CFTE_Dec(nn.Module):
    def __init__(self, N=64):
        super(CFTE_Dec, self).__init__()

        self.CFTE = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
            nn.ReLU(inplace=True),
            nn.Linear(N,16), 
        )

    def forward(self,mid, x_ref=None,use_delta=False):
        params = self.CFTE(mid).reshape(mid.size(0),4,4).unsqueeze(0)
        if x_ref is not None and use_delta:
            x_ref = x_ref['value']
            params = params+x_ref
        
        return {'value':params}

class FOMM_Enc(nn.Module):
    def __init__(self, N=64):
        super(FOMM_Enc, self).__init__()
    
        self.FOMM = nn.Sequential(
            nn.Linear(60,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
        )

    def forward(self,x, x_ref=None,use_delta=False):
        kp, jacobian = x['value'], x['jacobian']
        if x_ref is not None and use_delta: 
            kp_ref, jacobian_ref = x_ref['value'], x_ref['jacobian']
            kp, jacobian = kp-kp_ref, jacobian-jacobian_ref
        mid = self.FOMM(torch.cat([kp.reshape(kp.size(0),20),jacobian.reshape(kp.size(0),40)],dim=1))
        return mid      

class FOMM_Dec(nn.Module):
    def __init__(self, N=64):
        super(FOMM_Dec, self).__init__()
    
        self.FOMM = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,60),   
        )

    def forward(self,mid, x_ref=None,use_delta=False):
        
        kp, jacobian = self.FOMM(mid).split([20,40],dim=1)
        kp = kp.reshape(kp.size(0),10,2)
        jacobian = jacobian.reshape(kp.size(0),10,2,2)
        if x_ref is not None and use_delta:
            kp_ref, jacobian_ref = x_ref['value'], x_ref['jacobian']
            kp = kp+kp_ref
            jacobian = jacobian+jacobian_ref
        out = {
            'value': kp,
            'jacobian': jacobian,
        }
        return out


class FV2V_Enc(nn.Module):
    def __init__(self, N=64):
        super(FV2V_Enc, self).__init__()
    
        self.FV2V = nn.Sequential(
            nn.Linear(57,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
        )

    def forward(self, x,x_ref=None,use_delta=False):
        kp, rot, trans = x['exp'], x['rot_mat'], x['t']
        if x_ref is not None and use_delta:
            yaw_source = headpose_pred_to_degree(x_ref['yaw'])
            pitch_source = headpose_pred_to_degree(x_ref['pitch'])
            roll_source = headpose_pred_to_degree(x_ref['roll'])
            rot_ref = get_rotation_matrix(yaw_source, pitch_source, roll_source) 
            kp_ref, trans_ref = x_ref['exp'], x_ref['t']
            kp, rot, trans = kp-kp_ref, rot-rot_ref, trans-trans_ref
            
        mid = self.FV2V(torch.cat([kp,rot.reshape(kp.size(0),9),trans],dim=1))
        return mid


class FV2V_Dec(nn.Module):
    def __init__(self, N=64):
        super(FV2V_Dec, self).__init__()

        self.FV2V = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,57),   
        )    

    def forward(self,mid, x_ref=None,use_delta=False):
        kp, rot, trans = self.FV2V(mid).split([45,9,3],dim=1)
        rot = rot.reshape(mid.size(0),3,3)
        if x_ref is not None and use_delta:
            yaw_source = headpose_pred_to_degree(x_ref['yaw'])
            pitch_source = headpose_pred_to_degree(x_ref['pitch'])
            roll_source = headpose_pred_to_degree(x_ref['roll'])
            rot_ref = get_rotation_matrix(yaw_source, pitch_source, roll_source)
            kp_ref, trans_ref = x_ref['exp'], x_ref['t'].squeeze(1)
            rot = rot+rot_ref
      
            trans = trans+trans_ref
            kp = kp+kp_ref
  
        out = {'rot_mat':rot,
                't':trans,
                'exp':kp}
        return out  


class DAC_Enc(nn.Module):
    def __init__(self, N=64):
        super(DAC_Enc, self).__init__()
    
        self.DAC = nn.Sequential(
            nn.Linear(20,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,N),   
        )

    def forward(self,kp, kp_ref=None,use_delta=False):
        kp = kp['value']
        if kp_ref is not None and use_delta:
            kp_ref = kp_ref['value']
            kp = kp-kp_ref
        mid = self.DAC(kp.reshape(kp.size(0),20))

        return mid      

class DAC_Dec(nn.Module):
    def __init__(self, N=64):
        super(DAC_Dec, self).__init__()
    
        self.DAC = nn.Sequential(
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N),
            nn.ReLU(inplace=True),
            nn.Linear(N,N), 
            nn.ReLU(inplace=True),
            nn.Linear(N,20),   
        )

    def forward(self,mid, kp_ref=None,use_delta=False):
        kp = self.DAC(mid)
        kp = kp.reshape(kp.size(0),10,2)
        if kp_ref is not None and use_delta:
            kp_ref = kp_ref['value']
            kp = kp+kp_ref
        
        return {'value':kp}

