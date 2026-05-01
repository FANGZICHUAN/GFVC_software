import matplotlib
matplotlib.use('Agg')
import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from GFVC.utils import *
from GFVC.CFTE_utils import *
from GFVC.FOMM_utils import *
from GFVC.FV2V_utils import *
from GFVC.HDAC.bl_codecs import bl_decoders
from GFVC.codecs_utils import *
from GFVC.Translator.Paramtranscoder import *
from argparse import ArgumentParser
from skimage.transform import resize

from copy import copy

if __name__ == "__main__":
    #######Get options###########################################################################################################################################################         
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--original_seq", default='./testing_sequence/001_256x256.rgb', type=str, help="path to the input testing sequence")
    parser.add_argument("--encoding_frames", default=250, type=int, help="the number of encoding frames")
    parser.add_argument("--seq_width", default=256, type=int, help="the width of encoding frames")
    parser.add_argument("--seq_height", default=256, type=int, help="the height of encoding frames")
    parser.add_argument("--ae_qf", action='append', help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--iframe_qp", default=42, help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--iframe_format", default='YUV420', type=str,help="the quantization parameters for encoding the Intra frame")
    parser.add_argument("--device", default='cuda', type=str,help="execution device: [cpu, cuda]")
    parser.add_argument("--encoder_type", default='CFTE', type=str,help="encoder type of GFVC")

    parser.add_argument("--feature_coding_mode", default='AE', type=str,help="the implementation algorithm for supplement enhancement informantion on the basis of GFVC software")
    parser.add_argument("--sei_coordinate_qf", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--sei_matrix_qf", default=4, type=int, help="the quantization factor for the residual conversion from float-type to int-type")
    parser.add_argument("--use_base_layer", default='OFF', type=str,help="Flag to use hybrid coding framework (OFF sets animation-only reconstruction)")
    parser.add_argument("--bl_qp", default=50, type=int,help="QP value for encoding the base layer")    
    parser.add_argument("--no_analysis_in_decoder", default='OFF', type=str,help="QP value for encoding the base layer")
    parser.add_argument("--color_calibration", default='OFF', type=str,help="color_calibration")
    opt = parser.parse_args()
    ################################################################################################################################################################################
    ######Setting Options###########################################################################################################################################################
    #Number of frames to code
    frames=int(opt.encoding_frames)
    #Input Frame dimensions
    width=opt.seq_width
    height=opt.seq_height
    #Feature quantization factor for arithmetic coding
    q_step = list(map(int, opt.ae_qf))
    #Reference frame QP
    qp=opt.iframe_qp
    #Reference frame color format
    iframe_format=opt.iframe_format
    #Sequence number
    seq = os.path.splitext(os.path.split(opt.original_seq)[-1])[0]
    device = opt.device
    # GFVC type of encoder
    encoder_type = opt.encoder_type
    #Feature coding mode, AE or SEI
    feature_coding_mode=opt.feature_coding_mode
    #Quantization factor for coordinate in SEI
    coordinate_Qstep=opt.sei_coordinate_qf
    #Quantization factor for matrix in SEI
    matrix_Qstep=opt.sei_matrix_qf
    #Execution device
    if not torch.cuda.is_available():
        device = 'cpu'
    # print('Device:',device)
    #Base layer for HDAC model
    use_base_layer = (opt.use_base_layer=='ON')
    bl_qp = opt.bl_qp # ["37","42","47","52"]
    ## CFTE
    model_name = opt.model_name
    #Coding first frame for no analysis decoding
    no_analysis_in_decoder = (opt.no_analysis_in_decoder=='ON')
    if no_analysis_in_decoder:
        assert feature_coding_mode=='AE', 'No analysis in decoder is only supported for AE feature coding mode'
        no_analysis_in_decoder_flag = '+no_analysis_decoding'
    else:
        no_analysis_in_decoder_flag = ''
    color_calibration = (opt.color_calibration=='ON')
    ################################################################################################################################################################################
    #######Setting decoding translation if the encoder does not match###############################################################################################################################################
    model_dirname='../experiment/'+encoder_type+"/"+'Iframe_'+str(iframe_format) +"/"+ str(width)+'x'+str(height) +"/" +feature_coding_mode + no_analysis_in_decoder_flag 
    if encoder_type != model_name:
        translated =True
        assert feature_coding_mode == 'AE', 'Feature translation only supported for AE'
    else:
        translated = False
    if translated:
        print('--Decoding with feature translation--')
        trans_enc, trans_dec, encoder_analyzer = get_translator(encoder_type, model_name, device, trans_channel = 256)
        
        dec_output_path=os.path.join(model_dirname,model_name+'_dec') + '/'
    else:
      
        dec_output_path=os.path.join(model_dirname,'dec') + '/'
        
    
    ################################################################################################################################################################################   
    ######Setting directories#######################################################################################################################################################
    ## Input paths [compact features and reference frame bitstreams]
    #Base layer for HDAC
    if use_base_layer:
        cf_input_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        ref_input_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))+ '/'
        bl_input_path =os.path.join(model_dirname,'bl',seq+'_qp'+str(qp)+'_bqp'+str(bl_qp))
    else:
        cf_input_path = os.path.join(model_dirname,'kp',seq+'_qp'+str(qp)) + '/'
        ref_input_path = os.path.join(model_dirname,'enc',seq+'_qp'+str(qp))  + '/'
    # Output paths for decoded sequences
    
    os.makedirs(dec_output_path,exist_ok=True)     # the real decoded video  
    if use_base_layer:
        output_video_rgb =dec_output_path+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'.rgb'
        output_video_mp4 =dec_output_path+seq+'_qp'+str(qp)+'_bqp'+str(bl_qp)+'.mp4'
    
    else:
        output_video_rgb =dec_output_path+seq+'_qp'+str(qp)+'.rgb'
        output_video_mp4 =dec_output_path+seq+'_qp'+str(qp)+'.mp4'
    
    model_config_path = f'./GFVC/{model_name}/checkpoint/{model_name}.yaml'
    model_checkpoint_path = f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar'     
    f_dec=open(output_video_rgb,'w') 
    ################################################################################################################################################################################
    #######Codec and data preparation###############################################################################################################################################
    #Reference Image Decoder
    ref_decoder = RefereceImageDecoder(ref_input_path,qp,width=width,height=height,dec_name='vtm')
    # Keypoint Decoder
    if translated:
        kp_decoder = get_inter_coder(encoder_type)(model_config_path=f'./GFVC/{encoder_type}/checkpoint/{encoder_type}.yaml' if encoder_type=='FV2V' else None , model_checkpoint_path=f'./GFVC/{encoder_type}/checkpoint/{encoder_type}-checkpoint.pth.tar' if encoder_type=='FV2V' else None, kp_output_dir=cf_input_path, device=device) # q_step must set as default for feature translation
    else:
        kp_decoder = get_inter_coder(model_name)(
    model_config_path=f'./GFVC/{model_name}/checkpoint/{model_name}.yaml' if model_name in ['FV2V', 'TPSMM'] else None, 
    model_checkpoint_path=f'./GFVC/{model_name}/checkpoint/{model_name}-checkpoint.pth.tar' if model_name in ['FV2V', 'TPSMM'] else None,
    kp_output_dir=cf_input_path,
    q_step=q_step,
    device=device
)
    #Reconstruction models
    inter_coder = get_inter_coder(model_name)(model_config_path, model_checkpoint_path, device=device)
    out_video = []
    ################################################################################################################################################################################
    #######AE feature coding mode###################################################################################################################################################

    if use_base_layer:
        #create base layer spanning the entire sequence
        print('Creating base layer..')
        bl_decoding_params = {
                'fps':25,
                'frame_dim': (int(height),int(width)),
                'bin_path': bl_input_path}
        bl_dec = bl_decoders['vvc'](**bl_decoding_params)
        info_out = bl_dec.run()
        bl_frames = np.transpose(info_out['dec_frames'],[0,3,1,2])
   
        inter_coder.base_layer = bl_frames

    if feature_coding_mode=='AE':
        decoding_start_time = time.time()
        for frame_idx in tqdm(range(0, frames)):       
            with torch.no_grad():     
                if frame_idx in [0]:      # I-frame                      
                    reference, ref_bits = ref_decoder.decompress(frame_idx)  
                    
                         
                    if isinstance(reference, np.ndarray):
                        #convert to tensor
                        out_fr = reference
                        img_rec_out = np.transpose(reference,[2,0,1])
                        #convert the HxWx3 (uint8)-> 1x3xHxW (float32) 
                        reference = frame2tensor(img_rec_out)
                    else:
                        #When using LIC for reference compression we get back a tensor 1x3xHXW
                        img_rec_out = tensor2frame(reference)
                        out_fr = np.transpose(img_rec_out,[1,2,0])
                        reference = frame2tensor(img_rec_out)   # normalize to 0-1 
                
                    out_video.append(out_fr)
                    reference = reference.to(device)
                    if color_calibration:
                        
                        mean_ref = reference.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)
                        std_ref = reference.std(dim=(2, 3)).unsqueeze(2).unsqueeze(3)

                    img_rec_out.tofile(f_dec)
                     
                    #update decoder with the reference frame info
                    inter_coder.reference_frame_(reference)
                    kp_decoder.reference_frame_(reference)

                    if not no_analysis_in_decoder:
                        kp_reference = inter_coder.analysis_model(reference) 
                
                        # print(kp_reference)
                        if translated:
                            kp_reference_enc = encoder_analyzer(reference)
                        else:
                            kp_reference_enc = kp_reference
                    
                        inter_coder.kp_reference_(kp_reference)
                      
                        #append to list for use in predictively coding the next frame KPs
                        #KPs used for relative decoding of inter_Frame KPs
                        kp_value_frame = kp_decoder.get_kp_list(kp_reference_enc, frame_idx)
                        kp_decoder.rec_sem.append(kp_value_frame)
                    else:
                        dec_kp_ref_frame, kp_bits = kp_decoder.decode_kp(frame_idx)
                        
                        if translated:
                            dec_kp_ref_frame = trans_dec(trans_enc(dec_kp_ref_frame, use_delta=True))
                        inter_coder.kp_reference_(dec_kp_ref_frame)
                    Iframe_decoding_time = time.time() - decoding_start_time
                else:
                    # Decoding motion features
                    dec_kp_inter_frame, kp_bits = kp_decoder.decode_kp(frame_idx)
                    # print(dec_kp_inter_frame)
                    # Inter_frame reconstruction through animation
                
                    if translated:
                        dec_kp_inter_frame = trans_dec(trans_enc(dec_kp_inter_frame, kp_reference_enc, True), kp_reference, True)
                    # print(dec_kp_inter_frame)
                    predicted_frame  = inter_coder.predict_inter_frame(dec_kp_inter_frame, frame_idx)
                    if color_calibration:
                        mean_inter = predicted_frame.mean(dim=(2, 3)).unsqueeze(2).unsqueeze(3)
                        std_inter = predicted_frame.std(dim=(2, 3)).unsqueeze(2).unsqueeze(3)
                    

                        predicted_frame_normalized = (predicted_frame - mean_inter) / (std_inter + 1e-8)
                        predicted_frame = (predicted_frame_normalized * std_ref + mean_ref).clamp(0, 1)
                                    
                    pred = tensor2frame(predicted_frame)
                    #Save to output file
                    pred.tofile(f_dec)             
                    #HxWx3 format
                    out_video.append(np.transpose(pred,[1,2,0]))  
        total_decoding_time = time.time() - decoding_start_time
        print(f'Decoding time: {total_decoding_time:.2f}s, I-frame decoding time: {Iframe_decoding_time:.2f}s, p-frame decoding time: {(total_decoding_time-Iframe_decoding_time):.2f}s')

        f_dec.close()     
         
    ################################################################################################################################################################################
    #######SEI feature coding mode###################################################################################################################################################
    elif feature_coding_mode=='SEI' :
        decoding_start_time = time.time()
        dir_outSEI=model_dirname+'/outSEI/'+seq+'_qp'+str(qp)+'/'   
        os.makedirs(dir_outSEI,exist_ok=True)     # the real decoded video              


        outSEI=dir_outSEI+'generative_face_video.txt'
    
        if iframe_format=='RGB444':
            os.system("./image_codecs/vtm/decode_rgb444_SEI.sh "+ref_input_path+'allframe_org'+" "+outSEI)
        elif iframe_format=='YUV420':
            os.system("./image_codecs/vtm/decode_SEI.sh "+ref_input_path+'allframe_org'+" "+outSEI)

        # # 调用示例        
        split_text_file(outSEI, dir_outSEI)

        if iframe_format=='RGB444':
            
            ## Base Picture with Base matric reference
            f_temp=open(ref_input_path+'allframe_org_dec.rgb','rb')
            img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB
            out_video.append(img_rec)
        elif iframe_format=='YUV420':
            rec_ref_yuv=yuv420_to_rgb444(ref_input_path+'allframe_org_dec.yuv', width, height, 0, 1, False, False)
            img_rec = rec_ref_yuv[0]
            out_video.append(img_rec)
            img_rec = img_rec.transpose(2, 0, 1)    # HxWx3 -> 3xHxW
        img_rec.tofile(f_dec)
        img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1
        with torch.no_grad():
            reference = torch.tensor(img_rec[np.newaxis]).float() #.astype(np.float32)
            reference = reference.to(device)    # use configured device
            inter_coder.reference_frame_(reference)
            ref_kp_value = inter_coder.analysis_model(reference) 
            inter_coder.kp_reference_(ref_kp_value)   
            
            for frame in range(1,frames):
                kp_value = sei_decode(open(dir_outSEI+'gfv_'+str(frame)+'.txt', 'r', encoding='utf-8'), model_name ,device)
                # print(kp_value)
                prediction = inter_coder.predict_inter_frame(kp_value, frame)  #######################
                pred = tensor2frame(prediction)
                out_video.append(np.transpose(pred,[1,2,0])) 
                pred.tofile(f_dec)                        
        ###
        bin_save=ref_input_path+'allframe_org.bin'
        f_dec.close()     
        total_decoding_time = time.time() - decoding_start_time
        print(f'Decoding time: {total_decoding_time:.2f}s')  
             
    else:
        raise ValueError('Invalid feature coding mode. Please select either AE or SEI')

    imageio.mimsave(output_video_mp4,out_video, fps=25.0)

    # Clear CUDA cache to prevent memory issues in subsequent iterations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


