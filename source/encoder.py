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
from GFVC.keyframe_selector import SmartKeyframeSelector, save_keyframe_metadata
from GFVC.HDAC.bl_codecs import bl_encoders
from argparse import ArgumentParser
from skimage.transform import resize

from typing import Dict
from copy import copy
import time
    
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
    parser.add_argument("--smart_keyframe", default='OFF', type=str,help="使用智能关键帧选择 (ON/OFF)")
    parser.add_argument("--num_keyframes", default=1, type=int, help="智能关键帧数量，默认为1")
    opt = parser.parse_args()
    ################################################################################################################################################################################
    ######Setting Options###########################################################################################################################################################
    #Number of frames to code
    frames=int(opt.encoding_frames)
    #SourceWidth
    width = opt.seq_width 
    #SourceHeight
    height = opt.seq_height
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
    
    #智能关键帧选择
    use_smart_keyframe = (opt.smart_keyframe=='ON')
    #关键帧数量参数（从命令行获取，默认1）
    num_keyframes = int(opt.num_keyframes)
    print(f"[Encoder] 智能关键帧: {use_smart_keyframe}, 关键帧数量: {num_keyframes}")
    ################################################################################################################################################################################
    ######Setting directories#######################################################################################################################################################
    # Output directory
    model_dirname=f'../experiment/'+model_name+"/"+'Iframe_'+str(iframe_format) +"/"+ str(width)+'x'+str(height) +"/" + feature_coding_mode+no_analysis_in_decoder_flag +"/"       
    #Initialize the codec with models
    # Map LVPT to LivePortrait for file paths
    model_dir_name = 'LivePortrait' if model_name == 'LVPT' else model_name
    model_config_path = f'./GFVC/{model_dir_name}/checkpoint/{model_dir_name}.yaml'
    # LivePortrait uses 4 separate .pth files in checkpoint directory
    if model_name == 'LVPT' or model_name == 'LivePortrait':
        model_checkpoint_path = f'./GFVC/{model_dir_name}/checkpoint/'
    else:
        model_checkpoint_path = f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar' 
    #Base layer for HDAC
    if use_base_layer:
        kp_output_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        enc_output_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        bl_output_path =os.path.join(model_dirname,'bl',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))
        os.makedirs(bl_output_path,exist_ok=True)  
    else:
        # The feature to be compressed by GFVC model
        kp_output_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp))+ '/'
        # The feature to be compressed by vtm    
        enc_output_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp))+ '/'
    os.makedirs(enc_output_path,exist_ok=True)  
    os.makedirs(kp_output_path,exist_ok=True)    

    ################################################################################################################################################################################
    #######Codec and data preparation###############################################################################################################################################
    # Read the video sequence
    listR,listG,listB=raw_reader_planar(original_seq, width, height,frames)
    #Initialize the GFVC model
    inter_coder = get_inter_coder(model_name)(model_config_path, model_checkpoint_path, kp_output_path,q_step, device=device)
    
    #######智能关键帧选择##########################################################################################################################################################
    keyframe_list = [0]  # 默认第0帧为关键帧
    if use_smart_keyframe:
        print("="*60)
        print(f"启用智能关键帧选择 (数量: {num_keyframes})...")
        print("="*60)
        selector = SmartKeyframeSelector(inter_coder.analysis_model, device=device, num_keyframes=num_keyframes)
        keyframe_result = selector.select_keyframe(listR, listG, listB, width, height, frames)
        
        # 处理返回值可能是单个索引或列表
        if isinstance(keyframe_result, list):
            keyframe_list = keyframe_result
        else:
            keyframe_list = [keyframe_result]
        
        save_keyframe_metadata(keyframe_list, kp_output_path)
        print(f"智能关键帧选择完成: {keyframe_list} 将作为 I-frames")
        print("="*60)
    
    keyframe_set = set(keyframe_list)  # 用于快速查找
    
    #Reference image coder [VTM]
    ref_coder = ReferenceImageCoder(enc_output_path,qp,iframe_format,width,height,codec_name='vtm', device=device)
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
        Iframe_encoding_time = 0
        
        # 对于多关键帧，预先编码所有非第0帧的关键帧（设置参考）
        if use_smart_keyframe and len(keyframe_list) > 1:
            for kf_idx in keyframe_list:
                if kf_idx != 0:  # 第0帧会在主编码循环中处理
                    print(f"预编码关键帧 (第 {kf_idx} 帧) 用于设置参考...")
                    with torch.no_grad():
                        current_frame = [listR[kf_idx],listG[kf_idx],listB[kf_idx]]
                        reference, ref_bits = ref_coder.compress(current_frame, kf_idx)
                        
                        if isinstance(reference, np.ndarray):
                            reference = frame2tensor(np.transpose(reference,[2,0,1]))
                        reference = reference.to(device)
                        
                        kp_reference = inter_coder.analysis_model(reference)
                        kp_value_frame = inter_coder.get_kp_list(kp_reference, kf_idx, save=True)
                        
                        # 存储这个关键帧的信息，后续在主编码循环中使用
                        if not hasattr(inter_coder, 'preencoded_keyframes'):
                            inter_coder.preencoded_keyframes = {}
                        inter_coder.preencoded_keyframes[kf_idx] = {
                            'reference': reference,
                            'kp_reference': kp_reference,
                            'kp_value_frame': kp_value_frame
                        }
                        
                        if no_analysis_in_decoder:
                            rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_value_frame, kf_idx)
                        else:
                            inter_coder.rec_sem.append(kp_value_frame)
        elif use_smart_keyframe and 0 not in keyframe_list:
            # 单关键帧且不是第0帧的情况
            kf_idx = keyframe_list[0]
            print(f"编码智能选择的关键帧 (第 {kf_idx} 帧)...")
            with torch.no_grad():
                current_frame = [listR[kf_idx],listG[kf_idx],listB[kf_idx]]
                reference, ref_bits = ref_coder.compress(current_frame, kf_idx)
                
                if isinstance(reference, np.ndarray):
                    reference = frame2tensor(np.transpose(reference,[2,0,1]))
                reference = reference.to(device)
                
                kp_reference = inter_coder.analysis_model(reference)
                kp_value_frame = inter_coder.get_kp_list(kp_reference, kf_idx, save=True)
                
                inter_coder.reference_frame_(reference)
                inter_coder.kp_reference_(kp_reference)
                
                if no_analysis_in_decoder:
                    rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_value_frame, kf_idx)
                else:
                    inter_coder.rec_sem.append(kp_value_frame)
        
        # 辅助函数：找到最近的关键帧
        def get_nearest_keyframe_idx(frame_idx, keyframe_list):
            """找到小于等于frame_idx的最大关键帧索引"""
            valid_keyframes = [kf for kf in keyframe_list if kf <= frame_idx]
            return max(valid_keyframes) if valid_keyframes else keyframe_list[0]
        
        # 主编码循环 - 处理所有帧
        for frame_idx in tqdm(range(0, frames)):       
            with torch.no_grad():
                current_frame = [listR[frame_idx],listG[frame_idx],listB[frame_idx]]
                
                if frame_idx in keyframe_set:  # I-frame (一个或多个关键帧)
                    # 如果是预编码的关键帧，直接使用存储的信息
                    if hasattr(inter_coder, 'preencoded_keyframes') and frame_idx in inter_coder.preencoded_keyframes:
                        # 从预编码缓存获取
                        kf_data = inter_coder.preencoded_keyframes[frame_idx]
                        reference = kf_data['reference']
                        kp_reference = kf_data['kp_reference']
                        kp_value_frame = kf_data['kp_value_frame']
                        
                        # 设置当前参考
                        inter_coder.reference_frame_(reference)
                        inter_coder.kp_reference_(kp_reference)
                        
                        if frame_idx == 0:
                            Iframe_encoding_time = time.time() - encoding_start_time
                    else:
                        # 正常编码关键帧
                        reference, ref_bits = ref_coder.compress(current_frame, frame_idx)  
                          
                        if isinstance(reference, np.ndarray):
                            reference = frame2tensor(np.transpose(reference,[2,0,1]))
                        reference = reference.to(device)
                        #Extract motion representation vectors [Keypoints | Compact features etc]
                        kp_reference = inter_coder.analysis_model(reference) ################ 
                        kp_value_frame = inter_coder.get_kp_list(kp_reference, frame_idx, save=True)
                        
                        #update enc main with reference frame info
                        inter_coder.reference_frame_(reference)
                        inter_coder.kp_reference_(kp_reference)
                        
                        if no_analysis_in_decoder:
                            rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_value_frame, frame_idx)
                        else:
                            inter_coder.rec_sem.append(kp_value_frame)
                        
                        if frame_idx == 0:
                            Iframe_encoding_time = time.time() - encoding_start_time

                else:  # P-frame
                    # 多关键帧模式下，为每个P帧找到最近的关键帧作为参考
                    if use_smart_keyframe and len(keyframe_list) > 1:
                        nearest_kf = get_nearest_keyframe_idx(frame_idx, keyframe_list)
                        # 如果最近的关键帧不是当前已设置的参考，则更新参考
                        if nearest_kf != frame_idx and hasattr(inter_coder, 'preencoded_keyframes') and nearest_kf in inter_coder.preencoded_keyframes:
                            kf_data = inter_coder.preencoded_keyframes[nearest_kf]
                            inter_coder.reference_frame_(kf_data['reference'])
                            inter_coder.kp_reference_(kf_data['kp_reference'])
                    
                    inter_frame = cv2.merge(current_frame)
                    inter_frame = resize(inter_frame, (width, height))[..., :3]
                    inter_frame = torch.tensor(inter_frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                    inter_frame = inter_frame.to(device)    # require GPU | Use the available device      
                    ###extracting motion representation
                    kp_inter_frame = inter_coder.analysis_model(inter_frame) ################
    
                    kp_frame = inter_coder.get_kp_list(kp_inter_frame, frame_idx, save=True)
                    #Encode to binary string
                    rec_kp_frame, kp_bits = inter_coder.encode_kp(kp_frame, frame_idx)
               
        total_encoding_time = time.time() - encoding_start_time
        print(f'Encoding time: {total_encoding_time:.2f}s, I-frame encoding time: {Iframe_encoding_time:.2f}s, p-frame encoding time: {(total_encoding_time-Iframe_encoding_time):.2f}s')
    ################################################################################################################################################################################
    #######SEI feature coding mode###################################################################################################################################################
    elif feature_coding_mode=='SEI':
        # Initialize directory and list for SEI coding
        sei_encoder = SEIEncoder(model_name, coordinate_Qstep, matrix_Qstep, frames)
        cfg_enc =model_dirname+'/SEI/cfg/'+seq+'_QP'+str(qp)+'/'
        os.makedirs(cfg_enc,exist_ok=True)     # the cfg file    
        encoding_start_time = time.time()
        for frame_idx in tqdm(range(0, frames)):    
            frame_idx_str = str(frame_idx).zfill(4)  
            interframe = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
            interframe = resize(interframe, (width, height))[..., :3]

            with torch.no_grad():
                interframe = torch.tensor(interframe[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                interframe = interframe.to(device)    # use configured device

                ###extraction
                kp_interframe = inter_coder.analysis_model(interframe) ################
                kp_value = inter_coder.get_kp_list(kp_interframe, frame_idx)
                sei_encoder.update(kp_value)       
            ########## Generative Face Video Config Defination
         
        sei_encoder.prepare_to_code()          
        CONFIG_FILE = cfg_enc+'generative_face_video.cfg' 
        with open(CONFIG_FILE,'w') as f:
            f.writelines(sei_encoder.cfgcontent_basicinfo)  
            f.writelines(sei_encoder.cfgcontent_coordinate_matrix_info) 

        if iframe_format=='RGB444':
            f_temp=open(enc_output_path+'allframe_org.rgb','w')
            img_input_rgb = cv2.merge([listR[0],listG[0],listB[0]])
            img_input_rgb = img_input_rgb.transpose(2, 0, 1)   # 3xHxW
            img_input_rgb.tofile(f_temp)
            f_temp.close()            
        
        elif iframe_format=='YUV420':
            f_temp=open(enc_output_path+'allframe_org.yuv','w')
            img_input_rgb = cv2.merge([listR[0],listG[0],listB[0]])
            img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
            img_input_yuv.tofile(f_temp)
            f_temp.close()          
        
        if iframe_format=='YUV420':
   
            try:
                os.system("./image_codecs/vtm/encode_SEI.sh "+enc_output_path+'allframe_org'+" "+qp+" "+str(width)+" "+str(height)+" "+CONFIG_FILE+" "+str(1))  
            except Exception as e:
                print("Command execution failed:", str(e))            

        if iframe_format=='RGB444':
   
            try:
                os.system("./image_codecs/vtm/encode_rgb444_SEI.sh "+enc_output_path+'allframe_org'+" "+qp+" "+str(width)+" "+str(height)+" "+CONFIG_FILE+" "+str(1))    
            except Exception as e:
                print("Command execution failed:", str(e))
    

        bin_file=enc_output_path+'allframe_org.bin'

        total_encoding_time = time.time() - encoding_start_time
        print(f'Encoding time: {total_encoding_time:.2f}s')
    else:
        raise ValueError('Invalid feature coding mode. Please select either AE or SEI')

    # Clear CUDA cache to prevent memory issues in subsequent iterations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

