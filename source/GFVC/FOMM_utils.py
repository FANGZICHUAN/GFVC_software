import yaml
import numpy as np
from typing import Dict, List
import torch
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *

from GFVC.FOMM.sync_batchnorm import DataParallelWithCallback
from GFVC.FOMM.modules.generator import OcclusionAwareGenerator ###
from GFVC.FOMM.modules.keypoint_detector import KPDetector


    
def load_fomm_checkpoints(config_path, checkpoint_path, device='cpu'):

    with open(config_path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator.to(device)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    kp_detector.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
 
    generator.load_state_dict(checkpoint['generator'],strict=True)
    kp_detector.load_state_dict(checkpoint['kp_detector'], strict = True) ####

    if device=='cuda':
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return kp_detector,generator

    
class FOMM:
    '''FOMM Encoder'''
    def __init__(self, model_config_path:str=None, model_checkpoint_path:str=None, kp_output_dir:str=None,q_step:List[int]=[64,16], device:str='cpu') -> None:
        self.device = device
        if model_checkpoint_path is not None:
            fomm_analysis_model, fomm_synthesis_model = load_fomm_checkpoints(model_config_path, model_checkpoint_path, device=device)
            self.analysis_model = fomm_analysis_model.to(device)
            self.synthesis_model = fomm_synthesis_model.to(device)
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
        # print(kp_value.reshape(-1).tolist())
        kp_value_list = str(kp_value_list)
        kp_value_list = "".join(kp_value_list.split())

        kp_jacobian=kp_frame['jacobian'] 
        kp_jacobian_list=kp_jacobian.tolist()
        kp_jacobian_list=str(kp_jacobian_list)
        kp_jacobian_list="".join(kp_jacobian_list.split()) 
        if save:
            with open(self.kp_output_dir+'/frame'+str(frame_idx)+'.txt','w')as f:
                f.write(kp_value_list)  
                f.write('\n'+kp_jacobian_list)  

        kp_value_frame=json.loads(kp_value_list)###20
        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
        kp_jacobian_frame=json.loads(kp_jacobian_list)  ###40
        kp_jacobian_frame= eval('[%s]'%repr(kp_jacobian_frame).replace('[', '').replace(']', ''))
        #List of floating point values representing the keypoints and jacobian matrices
        # print(kp_value_frame)
        # print(kp_jacobian_frame)
        kp_list=kp_value_frame+kp_jacobian_frame ###20+40

        return kp_list
    
    def encode_kp(self,kp_list:List[str], frame_idx:int):
        ### residual
        if frame_idx == 0:
            kp_difference = (np.array(kp_list)).tolist()
        else:
            kp_difference=(np.array(kp_list)-np.array(self.rec_sem[-1])).tolist()
        ## quantization
        #=======================================================================#
        kp1_difference=kp_difference[:20]
        kp1_difference=[i*self.q_step[0] for i in kp1_difference]
        kp2_difference=kp_difference[20:]
        kp2_difference=[i*self.q_step[1] for i in kp2_difference]
        kp_difference = kp1_difference+kp2_difference
        #=======================================================================#
        kp_difference= list(map(round, kp_difference[:]))
        bin_file=self.kp_output_dir+'/frame'+str(frame_idx).zfill(4)+'.bin'

        final_encoder_expgolomb(kp_difference,bin_file)     

        bits=os.path.getsize(bin_file)*8        

        #### decoding for residual
        res_dec = final_decoder_expgolomb(bin_file)
        res_difference_dec = data_convert_inverse_expgolomb(res_dec)   

        ### (i)_th frame + (i+1-i)_th residual =(i+1)_th frame
        #=======================================================================#
        res1_difference_dec=res_difference_dec[:20]
        res1_difference_dec=[i/self.q_step[0] for i in res1_difference_dec]
        res2_difference_dec=res_difference_dec[20:]
        res2_difference_dec=[i/self.q_step[1] for i in res2_difference_dec]
        res_difference_dec = res1_difference_dec+res2_difference_dec
        #=======================================================================#
        if frame_idx == 0:
            rec_semantics=(np.array(res_difference_dec)).tolist()
        else:
            rec_semantics=(np.array(res_difference_dec)+np.array(self.rec_sem[-1])).tolist()

        self.rec_sem.append(rec_semantics)
        return rec_semantics, bits
    
    def decode_kp(self, frame_idx: int)->None:
        frame_index=str(frame_idx).zfill(4)
        bin_file=self.kp_output_dir+'/frame'+frame_index+'.bin' 
        bits=os.path.getsize(bin_file)*8              
        kp_dec = final_decoder_expgolomb(bin_file)

        ## decoding residual
        kp_difference = data_convert_inverse_expgolomb(kp_dec)
        #=======================================================================#
        ## inverse quantization
        kp1_difference_dec=kp_difference[:20]
        kp1_difference_dec=[i/self.q_step[0] for i in kp1_difference_dec]
        kp2_difference_dec=kp_difference[20:]
        kp2_difference_dec=[i/self.q_step[1] for i in kp2_difference_dec]
        kp_difference_dec = kp1_difference_dec+kp2_difference_dec
        #=======================================================================#
        kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  
   
        if frame_idx == 0:
            kp_previous= [0. for _ in range(len(kp_difference_dec))]
        else:
            kp_previous= eval('[%s]'%repr(self.rec_sem[-1]).replace('[', '').replace(']', '').replace("'", ""))  
        # kp_previous √
        # kp_difference_dec √
        
        kp_integer,kp_value,kp_jacobian=listformat_kp_jocobi_FOMM(kp_previous, kp_difference_dec) #######
        # kp_integer √
        # kp_value √
        # kp_jacobian √
        
        self.rec_sem.append(kp_integer)
  
        kp_inter_frame={}
        kp_value=json.loads(kp_value)
        kp_current_value=torch.Tensor(kp_value).to(self.device)          
        kp_inter_frame['value']=kp_current_value  

        kp_jacobian=json.loads(kp_jacobian)
        kp_current_jacobian=torch.Tensor(kp_jacobian).to(self.device)          
        kp_inter_frame['jacobian']=kp_current_jacobian  
        
        return kp_inter_frame, bits

    def predict_inter_frame(self,kp_inter_frame:Dict[str, torch.Tensor], frame_idx:int)->torch.Tensor:
        
        out = self.synthesis_model(self.reference_frame,kp_inter_frame, self.kp_reference)
  
        return out['prediction']










