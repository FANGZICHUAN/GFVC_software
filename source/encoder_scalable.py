import matplotlib
matplotlib.use('Agg')
import os
import cv2
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from GFVC.utils import *
from GFVC.codecs_utils import *
from GFVC.HDAC.bl_codecs import bl_encoders
from argparse import ArgumentParser
from skimage.transform import resize

from typing import Dict
from copy import copy
import time
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
    
if __name__ == "__main__":
    #######Get options###########################################################################################################################################################
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=512,type=int, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=512,type=int, help="the height of encoding frames")
    parser.add_argument("--ae_qf", action='append', help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    parser.add_argument("--feature_coding_mode", default='AE', type=str,help="the implementation algorithm for supplement enhancement informantion on the basis of GFVC software")
    parser.add_argument("--sei_coordinate_qf", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--sei_matrix_qf", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--use_base_layer", default='OFF', type=str,help="Flag to use hybrid coding framework (OFF sets animation-only reconstruction)")
    parser.add_argument("--bl_qp", default=50, type=int,help="QP value for encoding the base layer")
    parser.add_argument("--no_analysis_in_decoder", default='OFF', type=str,help="QP value for encoding the base layer")
    parser.add_argument("--SRLR_feature_size", default=32, type=int, help="feature size factor for encoding the scalable layer")
    parser.add_argument("--scalable_base_model_name", default='CFTE', type=str, help="base model name")
    parser.add_argument("--scalable_feature_code_mode", default='AE_Traditional', type=str, help="scalable_feature_code_mode")
    parser.add_argument("--scalable_feature_diff_mode", default='ON', type=str, help="scalable_feature_diff_mode")

    opt = parser.parse_args()
    ################################################################################################################################################################################
    ######Setting Options###########################################################################################################################################################
    #Number of frames to code
    frames=int(opt.encoding_frames)    
    
    #SourceWidth
    width = opt.seq_width 
    #SourceHeight
    height = opt.seq_width 
    #Feature quantization factor for arithmetic coding
    q_step = list(map(int, opt.ae_qf))

    #Reference frame QP
    qp = opt.iframe_qp 
    #Reference frame color format
    iframe_format=opt.iframe_format 
    #path to video input video sequences
    original_seq=opt.original_seq 
    #Execution device
    device = opt.device
    if not torch.cuda.is_available():
        device='cpu'
    #Sequence number
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    #Feature coding mode, AE or SEI
    feature_coding_mode=opt.feature_coding_mode
    #Quantization factor for coordinate in SEI
    coordinate_Qstep=opt.sei_coordinate_qf
    #Quantization factor for matrix in SEI
    matrix_Qstep=opt.sei_matrix_qf
    #Base layer for HDAC model
    use_base_layer = (opt.use_base_layer=='ON')
    bl_qp = opt.bl_qp # ["37","42","47","52"]
    #GFVC model name
    model_name = opt.model_name
    #Coding first frame for no analysis decoding
    no_analysis_in_decoder = (opt.no_analysis_in_decoder=='ON')
    if no_analysis_in_decoder:
        assert feature_coding_mode=='AE', 'No analysis in decoder is only supported for AE feature coding mode'
        no_analysis_in_decoder_flag = '+no_analysis_decoding'
    else:
        no_analysis_in_decoder_flag = ''

    SRLR_feature_size=opt.SRLR_feature_size     
    
    scalable_base_model_name=opt.scalable_base_model_name
    scalable_feature_code_mode=opt.scalable_feature_code_mode
    scalable_feature_diff_mode=opt.scalable_feature_diff_mode
        ################################################################################################################################################################################
    ######Setting directories#######################################################################################################################################################
    # Output directory
    model_dirname=f'../experiment-scalable/'+scalable_base_model_name+'_'+model_name+"/"+'FeatureSize_'+str(SRLR_feature_size)+'/'+'Iframe_'+str(iframe_format) +"/"+ str(width)+'x'+str(height) +"/" + feature_coding_mode+no_analysis_in_decoder_flag +"/"+scalable_feature_code_mode+'/'+scalable_feature_diff_mode+'/'       
    
    #Initialize the codec with models
    base_model_config_path = f'./GFVC/{scalable_base_model_name}/checkpoint/{scalable_base_model_name}.yaml'
    base_model_checkpoint_path = f'./GFVC/{scalable_base_model_name}/checkpoint/{scalable_base_model_name}-checkpoint.pth.tar' 
    
    #Initialize the codec with scalable models
    scalable_model_config_path = f'./GFVC/{model_name}/checkpoint/Feature_{SRLR_feature_size}/{model_name}.yaml'
    scalable_model_checkpoint_path = f'./GFVC/{model_name}/checkpoint/Feature_{SRLR_feature_size}/{model_name}-checkpoint.pth.tar'     
    
    
    #Base layer for HDAC
    if use_base_layer:
        kp_output_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        enc_output_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        bl_output_path =os.path.join(model_dirname,'bl',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))
        os.makedirs(bl_output_path,exist_ok=True)  
    else:
        # The feature to be compressed by GFVC base model
        kp_output_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp))+ '/'
        # The feature to be compressed by vtm    
        enc_output_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp))+ '/'
        # The feature to be compressed by SRLR model
        SRLR_feature_output_path = os.path.join(model_dirname,'feature',seq+'_qp'+str(qp))+ '/'
        # The initial rec by base model
        inirec_output_path=os.path.join(model_dirname,'inirec') + '/'
        
    os.makedirs(enc_output_path,exist_ok=True)  
    os.makedirs(kp_output_path,exist_ok=True)    
    os.makedirs(SRLR_feature_output_path,exist_ok=True)    
    os.makedirs(inirec_output_path,exist_ok=True)    
    output_video_rgb =inirec_output_path+seq+'_qp'+str(qp)+'.rgb'
    output_video_mp4 =inirec_output_path+seq+'_qp'+str(qp)+'.mp4'
    
    f_dec=open(output_video_rgb,'w') 
    out_video = []

    
#     ################################################################################################################################################################################
    #######Codec and data preparation###############################################################################################################################################
    # Read the video sequence
    listR,listG,listB=raw_reader_planar(original_seq, width, height,frames)
    #Initialize the GFVC base model
    inter_coder = get_inter_coder(scalable_base_model_name)(base_model_config_path, base_model_checkpoint_path, kp_output_path,q_step, device=device)
    #Reference image coder [VTM]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name='vtm', device=device)
    
    #Initialize the GFVC scalable model
    inter_scalable_coder = get_inter_scalable_coder(model_name)(scalable_model_config_path, scalable_model_checkpoint_path, 
                                                                SRLR_feature_output_path, q_step, SRLR_feature_size, device=device)      
    ################################################################################################################################################################################
    
    #######AE feature coding mode###################################################################################################################################################

    if use_base_layer:
        #create base layer spanning the entire sequence
        print(f'Creating base layer..')
        gop = np.transpose(np.array([listR,listG,listB]),[1,2,3,0])
        N, h,w,_ = gop.shape
        bl_coding_params = {
                'qp':bl_qp,
                'sequence':gop,
                'fps':25,
                'frame_dim': (h,w),
                'log_path': bl_output_path
            }
        
        bl_enc = bl_encoders['vvc'](**bl_coding_params)
        info_out = bl_enc.run()
        bl_frames = np.transpose(info_out['dec_frames'],[0,3,1,2])
        inter_coder.base_layer = bl_frames
   
    if feature_coding_mode=='AE':
        encoding_start_time = time.time()
        for frame_idx in tqdm(range(0, frames)):       
            with torch.no_grad():
                current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
                if frame_idx == 0:      # I-frame      
                    reference, ref_bits = ref_coder.compress(current_frame, frame_idx)  
                      
                    if isinstance(reference, np.ndarray):
                        reference = frame2tensor(np.transpose(reference,[2,0,1]))
                    reference = reference.to(device)
                    #Extract motion representation vectors [Keypoints | Compact features etc]
                    kp_reference =  inter_coder.analysis_model(reference) ################ 
                    kp_value_frame = inter_coder.get_kp_list(kp_reference, frame_idx, save=True)
          
                    #update enc main with reference frame info
                    inter_coder.reference_frame_(reference)
                    inter_coder.kp_reference_(kp_reference)
                    if no_analysis_in_decoder:
                    #Encode to binary string
                        rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_value_frame, frame_idx)
                    else:
                    #append to list for use in predictively coding the next frame KPs
                        inter_coder.rec_sem.append(kp_value_frame)
                    
                    #Extract motion representation vectors [different granuarityies Compact features for scalable coding]
                    feature_reference =  inter_scalable_coder.analysis_model(reference) ################ 
                    feature_value_frame = inter_scalable_coder.get_kp_list(feature_reference, frame_idx, save=True)                    
                    #update enc main with reference frame info
                    inter_scalable_coder.kp_reference_(feature_reference)    
                    if no_analysis_in_decoder:
                    #Encode to binary string
                        rec_feature_frame, feature_bits = inter_scalable_coder.encode_kp(feature_value_frame, frame_idx)
                    else:
                    #append to list for use in predictively coding the next frame KPs
                        inter_scalable_coder.rec_sem.append(feature_value_frame) 

                    Iframe_encoding_time = time.time() - encoding_start_time

                else:
                    inter_frame = cv2.merge(current_frame)
                    inter_frame = resize(inter_frame, (width, height))[..., :3]
                    inter_frame = torch.tensor(inter_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    inter_frame = inter_frame.to(device)    # require GPU | Use the available device      
                    
                    ###extracting motion representation
                    kp_inter_frame = inter_coder.analysis_model(inter_frame) ################
                    kp_frame = inter_coder.get_kp_list(kp_inter_frame, frame_idx, save=True)
                    #Encode to binary string
                    rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_frame, frame_idx)
                    
                    #Reconstruct video at the encoder side via base model
                    if scalable_base_model_name=="CFTE":
                        kp_integervalue = rec_kp_frame['value'].tolist()
                        kp_integer = listformat_adaptive_cfte_single(kp_integervalue[0], 1, 4)
                        rec_kp_inter_frame={}
                        kp_value=json.loads(kp_integer)
                        rec_kp_inter_frame['value']=torch.Tensor(kp_value).to(device)              

                    elif scalable_base_model_name=="FOMM":
                        kp_integervalue = rec_kp_frame #.tolist()
                        kp_integer,kp_value,kp_jacobian=listformat_kp_jocobi_FOMM_single(kp_integervalue) #######                       
                        rec_kp_inter_frame={}
                        kp_value=json.loads(kp_value)
                        kp_current_value=torch.Tensor(kp_value).to(device)          
                        rec_kp_inter_frame['value']=kp_current_value  

                        kp_jacobian=json.loads(kp_jacobian)
                        kp_current_jacobian=torch.Tensor(kp_jacobian).to(device)          
                        rec_kp_inter_frame['jacobian']=kp_current_jacobian                          
  
                    elif scalable_base_model_name=="FV2V":
                        kp_integervalue = rec_kp_frame #.tolist()
                        kp_integer,kp_mat_value,kp_t_value,kp_exp_value=listformat_kp_mat_exp_fv2v_single(kp_integervalue) #######
                        rec_kp_inter_frame={}                  
                        kp_mat_value=json.loads(kp_mat_value)
                        kp_current_mat=torch.Tensor(kp_mat_value).to(device)          
                        rec_kp_inter_frame['rot_mat']=kp_current_mat  
                        kp_t_value=json.loads(kp_t_value)
                        kp_current_t=torch.Tensor(kp_t_value).to(device)          
                        rec_kp_inter_frame['t']=kp_current_t  
                        kp_exp_value=json.loads(kp_exp_value)
                        kp_current_exp=torch.Tensor(kp_exp_value).to(device)          
                        rec_kp_inter_frame['exp']=kp_current_exp  
                        
                    elif scalable_base_model_name=="DAC":
                        kp_integervalue = rec_kp_frame['value'].tolist()                    
                        kp_integer,kp_value=listformat_kp_DAC_single(kp_integervalue) #######
                        rec_kp_inter_frame={}
                        kp_value=json.loads(kp_value)
                        kp_current_value=torch.Tensor(kp_value).reshape((1,10,2)).to(device)          
                        rec_kp_inter_frame['value']=kp_current_value                          


                    predicted_inter_frame  = inter_coder.predict_inter_frame(rec_kp_inter_frame, frame_idx)
                    pred_inter = tensor2frame(predicted_inter_frame)
                    #Save to output file
                    pred_inter.tofile(f_dec)             
                    out_video.append(np.transpose(pred_inter,[1,2,0]))  
                    
                    if scalable_feature_code_mode=='AE_Traditional':
                        if scalable_feature_diff_mode=='ON':
                            ###extracting scalable representation
                            feature_inter_frame_ori = inter_scalable_coder.analysis_model(inter_frame) ################
                            feature_inter_frame_base = inter_scalable_coder.analysis_model(predicted_inter_frame) ################
                            feature_inter_frame={}
                            feature_inter_frame['residual']=feature_inter_frame_ori['residual']-feature_inter_frame_base['residual']
                            feature_frame = inter_scalable_coder.get_kp_list(feature_inter_frame, frame_idx, save=True)
                            #Encode to binary string
                            rec_feature_frame, feature_bits = inter_scalable_coder.encode_kp(feature_frame, frame_idx)
                        else:
                            ###extracting scalable representation
                            feature_inter_frame = inter_scalable_coder.analysis_model(inter_frame) ################
                            feature_frame = inter_scalable_coder.get_kp_list(feature_inter_frame, frame_idx, save=True)
                            #Encode to binary string
                            rec_feature_frame, feature_bits = inter_scalable_coder.encode_kp(feature_frame, frame_idx)
                            
                    elif scalable_feature_code_mode=='AE_Learning':

                        ###extracting scalable representation
                        feature_inter_frame_ori = inter_scalable_coder.analysis_model(inter_frame) ################
                        feature_inter_frame_base = inter_scalable_coder.analysis_model(predicted_inter_frame) ################
                        rec_feature_frame, feature_bits = inter_scalable_coder.encode_feature_learning(feature_inter_frame_ori, feature_inter_frame_base, frame_idx)      
                        
        total_encoding_time = time.time() - encoding_start_time
        print(f'Encoding time: {total_encoding_time:.2f}s, I-frame encoding time: {Iframe_encoding_time:.2f}s, p-frame encoding time: {(total_encoding_time-Iframe_encoding_time):.2f}s')
        
        

# # ###############################################################################################################################################################################
#     #######SEI feature coding mode###################################################################################################################################################
#     elif feature_coding_mode=='SEI':
#         # Initialize directory and list for SEI coding
#         sei_encoder = SEIEncoder(model_name, coordinate_Qstep, matrix_Qstep, frames)
#         cfg_enc =model_dirname+'/SEI/cfg/'+seq+'_QP'+str(qp)+'/'
#         os.makedirs(cfg_enc,exist_ok=True)     # the cfg file    
#         encoding_start_time = time.time()
#         for frame_idx in tqdm(range(0, frames)):    
#             frame_idx_str = str(frame_idx).zfill(4)  
#             interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
#             interframe = resize(interframe, (width, height))[..., :3]

#             with torch.no_grad(): 
#                 interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
#                 interframe = interframe.cuda()    # require GPU                  

#                 ###extraction
#                 kp_interframe = inter_coder.analysis_model(interframe) ################
#                 kp_value = inter_coder.get_kp_list(kp_interframe, frame_idx)
#                 sei_encoder.update(kp_value)       
#             ########## Generative Face Video Config Defination
         
#         sei_encoder.prepare_to_code()          
#         CONFIG_FILE = cfg_enc+'generative_face_video.cfg' 
#         with open(CONFIG_FILE,'w') as f:
#             f.writelines(sei_encoder.cfgcontent_basicinfo)  
#             f.writelines(sei_encoder.cfgcontent_coordinate_matrix_info) 

#         if iframe_format=='RGB444':
#             f_temp=open(enc_output_path+'allframe_org.rgb','w')
#             img_input_rgb = cv2.merge([listR[0],listG[0],listB[0]])
#             img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
#             img_input_rgb.tofile(f_temp)
#             f_temp.close()            
        
#         elif iframe_format=='YUV420':
#             f_temp=open(enc_output_path+'allframe_org.yuv','w')
#             img_input_rgb = cv2.merge([listR[0],listG[0],listB[0]])
#             img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
#             img_input_yuv.tofile(f_temp)
#             f_temp.close()          
        
#         if iframe_format=='YUV420':
   
#             try:
#                 os.system("./image_codecs/vtm/encode_SEI.sh "+enc_output_path+'allframe_org'+" "+qp+" "+str(width)+" "+str(height)+" "+CONFIG_FILE+" "+str(1))  
#             except Exception as e:
#                 print("Command execution failed:", str(e))            

#         if iframe_format=='RGB444':
   
#             try:
#                 os.system("./image_codecs/vtm/encode_rgb444_SEI.sh "+enc_output_path+'allframe_org'+" "+qp+" "+str(width)+" "+str(height)+" "+CONFIG_FILE+" "+str(1))    
#             except Exception as e:
#                 print("Command execution failed:", str(e))
    

#         bin_file=enc_output_path+'allframe_org.bin'

#         total_encoding_time = time.time() - encoding_start_time
#         print(f'Encoding time: {total_encoding_time:.2f}s')
#     else:
#         raise ValueError('Invalid feature coding mode. Please select either AE or SEI')
        
#     # imageio.mimsave(output_video_mp4,out_video, fps=25.0)

