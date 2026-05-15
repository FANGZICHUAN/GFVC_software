# +
# get file size in python
import os
import numpy as np
from argparse import ArgumentParser
from eval_utils.utils import read_config_file

def get_all_file(dir_path):
    global files
    for filepath in os.listdir(dir_path):
        tmp_path = os.path.join(dir_path,filepath)
        if os.path.isdir(tmp_path):
            get_all_file(tmp_path)
        else:
            files.append(tmp_path)
    return files

def calc_files_size(files_path):
    files_size = 0
    for f in files_path:
        files_size += os.path.getsize(f)
    return files_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c","--config", default="../config/fv2v.yaml", type=str, help="Path to codec configuration file")
    parser.add_argument("--dataset_config", type=str, default=None,help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)
    if opt.dataset_config is not None:
        dataset_args = read_config_file(opt.dataset_config)
        seqlist= dataset_args['seq_list'] 
        Sequence_dir=dataset_args['sequence_dir'] 
        feature_coding_mode = dataset_args['feature_coding_mode']
        gop_size = dataset_args.get('gop_size', 100)
        if 'encoder_type' in dataset_args:
            encoder_type = dataset_args['encoder_type']
        else:
            encoder_type = model_name
    else:
        seqlist= args['seq_list'] 
        Sequence_dir=args['sequence_dir'] 
        feature_coding_mode = args['feature_coding_mode']
        gop_size = args.get('gop_size', 100)
        if 'encoder_type' in args:
            encoder_type = args['encoder_type']
        else:
            encoder_type = model_name
        
    frames = args['num_frames']
    qplist= args['qp_list'] 

    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256
    
    testingdata_name=args['dataset']
    Model=args['codec_name']  ## 'FV2V' OR 'FOMM' OR 'CFTE'
    Iframe_format=args['iframe_format']   ## 'YUV420'  OR 'RGB444'
    no_analysis_in_decoder = (args['no_analysis_in_decoder']=='ON')

    if Model in ['HDAC', 'HDAC_Extension']:
        bl_qplist = args['base_layer_params']['qp']
    
    if Model in ['HDAC']:
        qp_iter = bl_qplist
    else:
        qp_iter = qplist


    input_bin_file_path='../experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode
    save_path='../experiment/'+str(Model)+'/Iframe_'+str(Iframe_format)+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+"/resultBit/"
    os.makedirs(save_path, exist_ok=True)


    totalResult=np.zeros((len(seqlist)+1,len(qp_iter)))

    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qp_iter):  
            overall_bits=0
            for frame_idx in range(0, frames):            

                frame_idx_str = str(frame_idx).zfill(4)   

                if Model in ['HDAC']:
                    if frame_idx % gop_size == 0:      # I-frame bitstream (multi-reference frame support) 
                        if feature_coding_mode == 'AE':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qplist[0])+'_bqp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'
                        elif feature_coding_mode == 'SEI':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'/'+'allframe_org.bin'
                        else:
                            raise ValueError('Invalid feature coding mode')
                        overall_bits=overall_bits+os.path.getsize(Iframepath)*8
                        blpath = input_bin_file_path+'/bl/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qplist[0])+'_bqp'+str(QP)+'.bin'
                        overall_bits=overall_bits+os.path.getsize(blpath)*8 
                        if no_analysis_in_decoder:
                            kfpath = input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qplist[0])+'_bqp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'
                            overall_bits=overall_bits+os.path.getsize(kfpath)*8
                        
                    else:  ## Feature bitstream
                        if feature_coding_mode == 'AE':
                            interpath=input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(qplist[0])+'_bqp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'      
                            overall_bits=overall_bits+os.path.getsize(interpath)*8 

                elif Model in ['HDAC_Extension']:
                    if frame_idx % gop_size == 0:      # I-frame bitstream (multi-reference frame support) 
                        if feature_coding_mode == 'AE':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'/'+'frame'+frame_idx_str+'.bin'
                        elif feature_coding_mode == 'SEI':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'/'+'allframe_org.bin'
                        else:
                            raise ValueError('Invalid feature coding mode')
                        overall_bits=overall_bits+os.path.getsize(Iframepath)*8 

                        blpath = input_bin_file_path+'/bl/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'.bin'
                        overall_bits=overall_bits+os.path.getsize(blpath)*8 

                        if no_analysis_in_decoder:
                            kfpath = input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'/'+'frame'+frame_idx_str+'.bin'
                            overall_bits=overall_bits+os.path.getsize(kfpath)*8
                    
                    else:  ## Feature bitstream
                        if feature_coding_mode == 'AE':
                            interpath=input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'_bqp'+str(bl_qplist[0])+'/'+'frame'+frame_idx_str+'.bin'      
                            overall_bits=overall_bits+os.path.getsize(interpath)*8 
                
                else:       
    
                    if frame_idx % gop_size == 0:      # I-frame bitstream (multi-reference frame support) 
                        if feature_coding_mode == 'AE':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'
                        elif feature_coding_mode == 'SEI':
                            Iframepath = input_bin_file_path+'/enc/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'/'+'allframe_org.bin'
                        overall_bits=overall_bits+os.path.getsize(Iframepath)*8 

                        if no_analysis_in_decoder:
                            kfpath = input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'
                            overall_bits=overall_bits+os.path.getsize(kfpath)*8
                        
                    else:  ## Feature bitstream
                        if feature_coding_mode == 'AE':
                            interpath=input_bin_file_path+'/kp/'+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25fps_8bit_444_qp'+str(QP)+'/'+'frame'+frame_idx_str+'.bin'      

                            overall_bits=overall_bits+os.path.getsize(interpath)*8 



    
            totalResult[seqIdx][qpIdx]=overall_bits       

            
            
    # summary the bitrate
    for qp in range(len(qp_iter)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)

    np.set_printoptions(precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength

    np.savetxt(save_path+testingdata_name+'_'+'resultBit.txt', totalResult, fmt = '%.5f')                    
            
