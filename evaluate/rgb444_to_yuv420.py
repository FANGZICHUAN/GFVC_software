# +
import numpy as np
import os
import sys
import glob
import cv2
import yaml
from argparse import ArgumentParser

def read_config_file(config_path):
    '''Simply reads a yaml configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def RawReader_planar(FileName, ImgWidth, ImgHeight, NumFramesToBeComputed):
    
    if not os.path.exists(FileName):
        raise FileNotFoundError(FileName)
    f   = open(FileName, 'rb')
    frames  = NumFramesToBeComputed
    width   = ImgWidth
    height  = ImgHeight
    data = f.read()
    f.close()
    frame_size = width * height * 3
    if len(data) != frames * frame_size:
        available_frames = len(data) // frame_size
        raise ValueError(
            f'{FileName} contains {available_frames} complete frames '
            f'({len(data)} bytes), expected {frames} frames '
            f'({frames * frame_size} bytes)'
        )
    data = [int(x) for x in data]

    data_list=[]
    n=width*height
    for i in range(0,len(data),n):
        b=data[i:i+n]
        data_list.append(b)
    x=data_list

    listR=[]
    listG=[]
    listB=[]
    for k in range(0,frames):
        R=np.array(x[3*k]).reshape((width, height)).astype(np.uint8)
        G=np.array(x[3*k+1]).reshape((width, height)).astype(np.uint8)
        B=np.array(x[3*k+2]).reshape((width, height)).astype(np.uint8)
        listR.append(R)
        listG.append(G)
        listB.append(B)
    return listR,listG,listB

def splitlist(list): 
    alist = []
    a = 0 
    for sublist in list:
        try: #用try来判断是列表中的元素是不是可迭代的，可以迭代的继续迭代
            for i in sublist:
                alist.append (i)
        except TypeError: #不能迭代的就是直接取出放入alist
            alist.append(sublist)
    for i in alist:
        if type(i) == type([]):#判断是否还有列表
            a =+ 1
            break
    if a==1:
        return printlist(alist) #还有列表，进行递归
    if a==0:
        return alist  

        
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="../config/cfte.yaml", type=str, help="Path to codec configuration file")
    parser.add_argument("--dataset_config", type=str, default=None,help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)
    if opt.dataset_config is not None:
        dataset_args = read_config_file(opt.dataset_config)
        seqlist= dataset_args['seq_list'] 
        Sequence_dir=dataset_args['sequence_dir'] 
        
    else:
        seqlist= args['seq_list'] 
        Sequence_dir=args['sequence_dir'] 
        

    frames = args['num_frames']
    testingdata_name=args['dataset']

    #'FV2V' OR 'FOMM' OR 'CFTE' ###You should choose which GFVC model to be used.
    Model=args['codec_name']
    
    #[ "22", "32", "42", "52"]
    qplist= args['qp_list'] 

    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256

    feature_coding_mode = args['feature_coding_mode']
    no_analysis_in_decoder = (args['no_analysis_in_decoder']=='ON')
    if no_analysis_in_decoder:
        assert feature_coding_mode=='AE', 'No analysis in decoder is only supported for AE feature coding mode'
        no_analysis_in_decoder_flag = '+no_analysis_decoding'
    else:
        no_analysis_in_decoder_flag = ''


    Iframe_format=args['iframe_format'] ## 'YUV420'  OR 'RGB444'

    if Model in ['HDAC', 'HDAC_Extension']:
        bl_qplist = args['base_layer_params']['qp']
    
    if 'encoder_type' in args:
        encoder_type = args['encoder_type']
    else:
        encoder_type = Model

    if encoder_type != Model:
        translated =True
    else:
        translated = False
    
    if translated:
        base_dir = '../experiment/'+encoder_type+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/'+Model+'_dec/'
        result_dir = '../experiment/'+encoder_type+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/'+Model+'_dec_yuv420/'
    else:
        base_dir = '../experiment/'+Model+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/dec/'
        result_dir = '../experiment/'+Model+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/dec_yuv420/'
    os.makedirs(result_dir,exist_ok=True)     

    
    for seq in seqlist:
        for qp in qplist:    
            if Model in ['HDAC', 'HDAC_Extension']:
                for blqp in bl_qplist:
                    original_seq= base_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qp)+'_bqp'+str(blqp)+'.rgb' 
                    listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

                    # wtite ref and cur (rgb444) to file (yuv420)
                    oriyuv_path=result_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_420_qp'+str(qp)+'_bqp'+str(blqp)+'.yuv' 
                    f_temp=open(oriyuv_path,'w')            
                    for frame_idx in range(0, frames):            

                        img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                        img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                        img_input_yuv.tofile(f_temp)
                    f_temp.close()   
            else:
                original_seq= base_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qp)+'.rgb' 
                listR,listG,listB=RawReader_planar(original_seq, width, height,frames)

                # wtite ref and cur (rgb444) to file (yuv420)
                
                oriyuv_path=result_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_420_qp'+str(qp)+'.yuv' 
                f_temp=open(oriyuv_path,'w')            
                for frame_idx in range(0, frames):            

                    img_input_rgb = cv2.merge([listR[frame_idx],listG[frame_idx],listB[frame_idx]])
                    img_input_yuv = cv2.cvtColor(img_input_rgb, cv2.COLOR_RGB2YUV_I420)  #COLOR_RGB2YUV
                    img_input_yuv.tofile(f_temp)
                f_temp.close()   