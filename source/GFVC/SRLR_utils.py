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
from GFVC.SRLR.sync_batchnorm import DataParallelWithCallback
from GFVC.SRLR.modules.util import *

from GFVC.SRLR.modules.face_residual import Residual_Detector
from GFVC.SRLR.modules.fusion import *
from GFVC.SRLR.modules.VideoCompressor import VideoCompressor ###
from GFVC.SRLR.modules.VideoDecompressor import VideoDecompressor ###



def load_srlr_checkpoints(config_path, checkpoint_path, device='cpu'):

    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
                
    residual_detector = Residual_Detector(**config['model_params']['kp_detector_params'], 
                                          **config['common_params'])    
    residual_detector.to(device)
        
    fusiongenerator = FusionGenerator(**config['model_params']['generator_params'], 
                                      **config['common_params'])
    fusiongenerator.to(device)
        
    videocompressor = VideoCompressor(**config['model_params']['videocompressor_params'])
    videodecompressor = VideoDecompressor(**config['model_params']['videocompressor_params'])
    videocompressor.to(device)          
    videodecompressor.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    fusiongenerator.load_state_dict(checkpoint['fusiongenerator'],strict=False) 
    residual_detector.load_state_dict(checkpoint['residual_detector'],strict=False) 
    videocompressor.load_state_dict(checkpoint['videocompressor'],strict=False) 
    videodecompressor.load_state_dict(checkpoint['videocompressor'],strict=False) 
    
    if device =='cuda':
        residual_detector = DataParallelWithCallback(residual_detector)
        fusiongenerator = DataParallelWithCallback(fusiongenerator)
    #     videocompressor = DataParallelWithCallback(videocompressor)
    #     videodecompressor = DataParallelWithCallback(videodecompressor)
    residual_detector.eval()
    fusiongenerator.eval()
    videocompressor.eval()
    videodecompressor.eval()
    return residual_detector, fusiongenerator, videocompressor, videodecompressor    


class SRLR:
    def __init__(self, model_config_path:str=None, model_checkpoint_path:str=None, kp_output_dir:str=None,q_step:List[int]=[4], feature_size:List[int]=[32],  device:str='cpu') -> None:
        self.device = device
        if model_checkpoint_path is not None:
            srlr_analysis_Model, srlr_synthesis_model, srlr_videocompressor, srlr_videodecompressor = load_srlr_checkpoints(model_config_path, model_checkpoint_path, device=device)
            
            self.analysis_model = srlr_analysis_Model
            self.synthesis_model = srlr_synthesis_model
            self.videocompressor = srlr_videocompressor
            self.videodecompressor = srlr_videodecompressor
            
            self.reference_frame = None
            self.kp_reference = None
        if q_step is not None:
            self.kp_output_dir = kp_output_dir
            self.q_step = q_step
            self.rec_sem = []
            self.feature_size = feature_size
    
    def reference_frame_(self,x):
        self.reference_frame = x
    
    def kp_reference_(self,x):
        self.kp_reference = x
            

    def filesize(self,filepath: str) -> int:
        if not Path(filepath).is_file():
            raise ValueError(f'Invalid file "{filepath}".')
        return Path(filepath).stat().st_size

    def write_bytes(self,fd, values, fmt=">{:d}s"):
        if len(values) == 0:
            return
        fd.write(struct.pack(fmt.format(len(values)), values))


    def read_bytes(self,fd, n, fmt=">{:d}s"):
        sz = struct.calcsize("s")
        return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

    
    def get_kp_list(self, kp_frame: Dict[str,torch.Tensor], frame_idx:int, save:bool=False)->List[str]:
        kp_value = kp_frame['residual']
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
        kp_inter_frame['residual']=kp_current_value  
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

        kp_integer = listformat_adaptive_cfte(kp_previous, kp_difference_dec, 1, self.feature_size)
        self.rec_sem.append(kp_integer)
  
        cf_inter_frame={}
        cf_value=json.loads(kp_integer)
        cf_current_value=torch.Tensor(cf_value).to(self.device)          
        cf_inter_frame['residual']=cf_current_value  
        return cf_inter_frame, bits

    
    def predict_inter_frame(self, base_gene_frame: torch.Tensor, feature_inter_frame: torch.Tensor, frame_idx:int=None)->torch.Tensor:
        out = self.synthesis_model(self.reference_frame, base_gene_frame, feature_inter_frame)
        return out['fused']

    
    def encode_feature_learning(self, kp_list_ori:List[str], kp_list_gene:List[str], frame_idx:int):
        ### residual

        bits=0
        output_ystrings=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'_ystrings.bin'
        output_zstrings=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'_zstrings.bin'                   

        y_strings, z_strings = self.videocompressor(kp_list_ori, kp_list_gene)  

        with open(output_ystrings,'wb') as f1:
            #print(len(fs_strings[0]))
            self.write_bytes(f1, y_strings[0])
        ### bit estimation
        size1 = self.filesize(output_ystrings)
        bits=bits+float(size1) * 8

        with open(output_zstrings,'wb') as f2:
            #print(len(fs_strings[0]))
            self.write_bytes(f2, z_strings[0])                            
        ### bit estimation
        size2 = self.filesize(output_zstrings)
        bits=bits+float(size2) * 8
      
     #######Decode 
        with open(output_ystrings,'rb') as fy:
            size_rec1 = self.filesize(output_ystrings)
            decode_ystrings = []
            y_rec = self.read_bytes(fy,size_rec1)
            decode_ystrings.append(y_rec)  

        with open(output_zstrings,'rb') as fz:
            size_rec2 = self.filesize(output_zstrings)
            decode_zstrings = []
            z_rec = self.read_bytes(fz,size_rec2)
            decode_zstrings.append(z_rec)  

        kp_inter_frame = self.videodecompressor(decode_ystrings, decode_zstrings, kp_list_gene) 

        # kp_inter_frame={}      
        # kp_inter_frame['residual']=rec_current_residual  
        #reconstruct the KPs 
        return kp_inter_frame, bits
    
    
    def decode_feature_learning(self, kp_list_gene:List[str], frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)

        bits=0
        output_ystrings=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'_ystrings.bin'
        output_zstrings=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'_zstrings.bin'                   

        ### bit estimation
        size1 = self.filesize(output_ystrings)
        bits=bits+float(size1) * 8
        
        ### bit estimation
        size2 = self.filesize(output_zstrings)
        bits=bits+float(size2) * 8
      
     #######Decode 
        with open(output_ystrings,'rb') as fy:
            size_rec1 = self.filesize(output_ystrings)
            decode_ystrings = []
            y_rec = self.read_bytes(fy,size_rec1)
            decode_ystrings.append(y_rec)  

        with open(output_zstrings,'rb') as fz:
            size_rec2 = self.filesize(output_zstrings)
            decode_zstrings = []
            z_rec = self.read_bytes(fz,size_rec2)
            decode_zstrings.append(z_rec)  

        kp_inter_frame = self.videodecompressor(decode_ystrings, decode_zstrings, kp_list_gene) 
        
        #reconstruct the KPs 
        return kp_inter_frame, bits