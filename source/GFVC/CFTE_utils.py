import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml

import numpy as np
import torch
from typing import Dict, List
import json
import torch.nn.functional as F
from pathlib import Path
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
from GFVC.CFTE.sync_batchnorm import DataParallelWithCallback
from GFVC.CFTE.modules.util import *
from GFVC.CFTE.modules.generator import OcclusionAwareGenerator ###
from GFVC.CFTE.modules.keypoint_detector import KPDetector ###

def load_cfte_checkpoints(config_path, checkpoint_path, device='cpu'):
    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict=True) ####

    if device =='cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator

class CFTE:
    def __init__(self, model_config_path:str=None, model_checkpoint_path:str=None, kp_output_dir:str=None,q_step:List[int]=[4], device:str='cpu') -> None:
        self.device = device
        if model_checkpoint_path is not None:
            cfte_analysis_Model, cfte_synthesis_model = load_cfte_checkpoints(model_config_path, model_checkpoint_path, device=device)
            self.analysis_model = cfte_analysis_Model
            self.synthesis_model = cfte_synthesis_model
            self.reference_frame = None
            self.kp_reference = None
        if q_step is not None:
            self.kp_output_dir = kp_output_dir
            self.q_step = q_step
            self.rec_sem = []
    
    def reference_frame_(self,x):
        self.reference_frame = x
    
    def kp_reference_(self,x):
        self.kp_reference = x
            

    
    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int, save:bool=False)->List[str]:
        kp_value = kp_frame['value']
        kp_value_list = kp_value.tolist()
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())
        if save:
            with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
                f.write(kp_value_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        return kp_value_frame
    
    def encode_kp(self,kp_list:List[str], frame_idx:int):
        ### residual
        if len(self.rec_sem) == 0:
            kp_difference = (np.array(kp_list)).tolist()
        else:
            kp_difference=(np.array(kp_list)-np.array(self.rec_sem[-1])).tolist()
        ## quantization

        kp_difference=[i*self.q_step[0] for i in kp_difference]
        kp_difference= list(map(round, kp_difference[:]))

        bin_file=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        kp_dec = final_decoder_expgolomb(bin_file)
        kp_difference_dec = data_convert_inverse_expgolomb(kp_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame

        kp_difference_dec=[i/self.q_step[0] for i in kp_difference_dec]

        if len(self.rec_sem) == 0:
            kp_integer,kp_value=listformat_kp_DAC([0. for _ in range(len(kp_difference_dec))], kp_difference_dec)
        else:
            kp_integer,kp_value=listformat_kp_DAC(self.rec_sem[-1], kp_difference_dec) #######
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).to(self.device)          
        kp_inter_frame['value']=kp_current_value  
        #reconstruct the KPs 
        return kp_inter_frame, bits
    
    def decode_kp(self, frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+frame_index+'.bin' 
        bits=os.path.getsize(bin_file)*8              
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        ## inverse quantization
        kp_difference_dec=[i/self.q_step[0] for i in kp_difference]
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        if len(self.rec_sem) == 0:
            kp_previous= [0. for _ in range(len(kp_difference_dec))]
        else:
            kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  

        kp_integer = listformat_adaptive_cfte(kp_previous, kp_difference_dec, 1, 4)
        self.rec_sem.append(kp_integer)
  
        cf_inter_frame={}
        cf_value=json.loads(kp_integer)
        cf_current_value=torch.Tensor(cf_value).to(self.device)          
        cf_inter_frame['value']=cf_current_value  
        return cf_inter_frame, bits

    
    
    def predict_inter_frame(self, kp_inter_frame: Dict[str,torch.Tensor], frame_idx:int=None)->torch.Tensor:
        out = self.synthesis_model(self.reference_frame, self.kp_reference, kp_inter_frame)
        return out['prediction']
        
    