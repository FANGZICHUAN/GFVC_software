import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml

import numpy as np
import torch
from typing import Dict, List
import json
import cv2
import torch.nn.functional as F

from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from GFVC.FV2V.sync_batchnorm import DataParallelWithCallback
from GFVC.FV2V.modules.generator import OcclusionAwareGenerator ###
from GFVC.FV2V.modules.keypoint_detector import KPDetector, HEEstimator



def load_fv2v_checkpoints(config_path, checkpoint_path, device='cpu'):

    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
        
    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    he_estimator.to(device)        
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'],strict=True) ####
    he_estimator.load_state_dict(checkpoint['he_estimator'],strict=True)
        
    if device=='cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()
    
    return kp_detector, he_estimator, generator



def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

# def get_rotation_matrix(yaw, pitch, roll):
#     yaw = yaw / 180 * 3.14
#     pitch = pitch / 180 * 3.14
#     roll = roll / 180 * 3.14

#     roll = roll.unsqueeze(1)
#     pitch = pitch.unsqueeze(1)
#     yaw = yaw.unsqueeze(1)
#     roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
#                           torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
#                           torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
#     roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)
#     pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
#                            torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
#                            -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
#     pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)
#     yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
#                          torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
#                          torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
#     yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)
#     rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

#     return rot_mat
#=======================================================================#
#For new multiRes
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat
#=======================================================================#
def keypoint_transformation(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']
    
    if 'rot_mat' in he:
        rot_mat = he['rot_mat']
    else:   
        if not free_view:
            yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
            yaw = headpose_pred_to_degree(yaw)
            pitch = headpose_pred_to_degree(pitch)
            roll = headpose_pred_to_degree(roll)
          
        else:
            if yaw is not None:
                yaw = torch.tensor([yaw]).cuda()
            else:
                yaw = he['yaw']
                yaw = headpose_pred_to_degree(yaw)
            if pitch is not None:
                pitch = torch.tensor([pitch]).cuda()
            else:
                pitch = he['pitch']
                pitch = headpose_pred_to_degree(pitch)
            if roll is not None:
                roll = torch.tensor([roll]).cuda()
            else:
                roll = he['roll']
                roll = headpose_pred_to_degree(roll)

        rot_mat = get_rotation_matrix(yaw, pitch, roll)
        # print(rot_mat)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)
    #print(kp_rotated.shape)
    
    
    t, exp = he['t'], he['exp']
    # keypoint translation

    t = t.unsqueeze_(1)
    t = t.repeat(1, kp.shape[1], 1)
    #t = t.repeat_interleave(kp.shape[1], dim=1)
    
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}


class FV2V:
    '''Wrapper for models and forward methods'''
    def __init__(self, model_config_path:str=None, model_checkpoint_path:str=None, kp_output_dir:str=None,q_step:List[int]=[256], device:str='cpu') -> None:
        self.device = device
        if model_checkpoint_path is not None:
            fv2v_analysis_model_detector, fv2v_analysis_model_estimator, fv2v_synthesis_model = load_fv2v_checkpoints(model_config_path, model_checkpoint_path, device=device)
            self.detector_model = fv2v_analysis_model_detector.to(self.device)
            # print(self.detector_model)
            self.analysis_model = fv2v_analysis_model_estimator.to(self.device)
            self.synthesis_model = fv2v_synthesis_model.to(self.device)

            self.reference_frame = None
            self.kp_canonical = None
            self.kp_reference = None
        if q_step is not None:
            self.kp_output_dir = kp_output_dir
            self.q_step = q_step
            self.rec_sem = []
    
    def reference_frame_(self,x):
        self.reference_frame = x
        self.kp_canonical = self.detector_model(x)

    
    def kp_reference_(self,he_source):
 
        self.kp_reference = keypoint_transformation(self.kp_canonical, he_source, estimate_jacobian=False,
                                                                free_view=False, yaw=0, pitch=0, roll=0)             

    
    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int, save:bool=False)->List[str]:

        yaw=kp_frame['yaw']
        pitch=kp_frame['pitch']
        roll=kp_frame['roll']

        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)

        rot_mat = get_rotation_matrix(yaw, pitch, roll)            
        rot_mat_list=rot_mat.tolist()
        rot_mat_list=str(rot_mat_list)
        rot_mat_list="".join(rot_mat_list.split())               

        t=kp_frame['t']
        t_list=t.tolist()
        t_list=str(t_list)
        t_list="".join(t_list.split())

        exp=kp_frame['exp']
        exp_list=exp.tolist()
        exp_list=str(exp_list)
        exp_list="".join(exp_list.split())   

        if save:
            with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
                f.write(rot_mat_list)
                f.write('\n'+t_list)  
                f.write('\n'+exp_list)

        rot_frame=json.loads(rot_mat_list)###torch.Size([1, 3, 3])
        rot_frame= eval('[%s]'%repr(rot_frame).replace('[', '').replace(']', ''))

        t_frame=json.loads(t_list)  ###torch.Size([1, 3])
        t_frame= eval('[%s]'%repr(t_frame).replace('[', '').replace(']', ''))

        exp_frame=json.loads(exp_list)  ###torch.Size([1, 45])
        exp_frame= eval('[%s]'%repr(exp_frame).replace('[', '').replace(']', ''))

        kp_integer=rot_frame+t_frame+exp_frame ###9+3+45=57
        return kp_integer  
    
    def encode_kp(self,kp_list:List[str], frame_idx:int):
        ### residual
        if frame_idx == 0:
            kp_difference = (np.array(kp_list)).tolist()
        else:
            kp_difference=(np.array(kp_list)-np.array(self.rec_sem[-1])).tolist()
        # print(kp_difference)
        #=======================================================================#
        kp_difference=[i*self.q_step[0] for i in kp_difference]
        #=======================================================================#
        kp_difference= list(map(round, kp_difference[:]))
        # print(kp_difference)

        bin_file=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)  
        # print(res_difference_dec) 

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        #=======================================================================#
        res_difference_dec=[i/self.q_step[0] for i in res_difference_dec]
        #=======================================================================#
        # print(res_difference_dec)
        if frame_idx == 0:
            rec_semantics=(np.array(res_difference_dec)).tolist()
        else:
            rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[-1])).tolist()

        # print(rec_semantics)
        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits
    
    def decode_kp(self,frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+frame_index+'.bin' 
        bits=os.path.getsize(bin_file)*8              
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quantization
        #=======================================================================#
        ## inverse quantization
        kp_difference_dec=[i/self.q_step[0] for i in kp_difference]

        #=======================================================================#
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        if frame_idx == 0:
            kp_previous= [0. for _ in range(len(kp_difference_dec))]
        else:
            kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer,kp_mat_value,kp_t_value,kp_exp_value=listformat_kp_mat_exp_fv2v(kp_previous, kp_difference_dec) #######
        self.rec_sem.append(kp_integer)


        rec_kp_current={}                  
        kp_mat_value=json.loads(kp_mat_value)
        kp_current_mat=torch.Tensor(kp_mat_value).to(self.device)          
        rec_kp_current['rot_mat']=kp_current_mat  

        kp_t_value=json.loads(kp_t_value)
        kp_current_t=torch.Tensor(kp_t_value).to(self.device)          
        rec_kp_current['t']=kp_current_t  

        kp_exp_value=json.loads(kp_exp_value)
        kp_current_exp=torch.Tensor(kp_exp_value).to(self.device)          
        rec_kp_current['exp']=kp_current_exp  

        return rec_kp_current, bits
        
    def predict_inter_frame(self, rec_he_current:Dict[str, torch.Tensor], frame_idx:int=None, relative=False,
                            adapt_movement_scale=False, estimate_jacobian=False, cpu=False, free_view=False, 
                            yaw=0, pitch=0, roll=0)->torch.Tensor:
      
        # print(rec_he_current)
        kp_current = keypoint_transformation(self.kp_canonical, rec_he_current, estimate_jacobian=False, 
                                                    free_view=False, yaw=0, pitch=0, roll=0)  
        # print(kp_current['value'])
        # print(self.kp_reference['value'])
        out = self.synthesis_model(self.reference_frame,kp_source=self.kp_reference,kp_driving= kp_current)
        # print(out['prediction'])
        return out['prediction']