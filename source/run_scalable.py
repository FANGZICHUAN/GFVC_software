# +
import os
from tqdm import tqdm
from argparse import ArgumentParser
from GFVC.utils import read_config_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='../config/srlr.yaml', type=str, help="Path to codec configuration file")
    parser.add_argument("--dataset_config", type=str, default=None,help="Path to codec configuration file")
    opt = parser.parse_args()
    args = read_config_file(opt.config)
    if opt.dataset_config is not None:
        dataset_args = read_config_file(opt.dataset_config)
        seqlist= dataset_args['seq_list'] 
        Sequence_dir=dataset_args['sequence_dir'] 
        feature_coding_mode = dataset_args['feature_coding_mode']
    else:
        seqlist= args['seq_list'] 
        Sequence_dir=args['sequence_dir'] 
        feature_coding_mode = args['feature_coding_mode']
        

    frames = args['num_frames']
    testingdata_name=args['dataset']
    model_name=args['codec_name']
    
    

    qplist= args['qp_list'] 

    bl_qplist = args['base_layer_params']['qp']
    use_base_layer = args['base_layer_params']['use_base_layer']
    
    #Input Frame dimensions
    height=args['height'] #256
    width=args['width'] #256
  
    coding_mode=args['inference_mode'] 
    iframe_format=args['iframe_format']  
    
    ae_qf_1=args['ae_qf'][0]
    ae_qf_2=args['ae_qf'][1]
    sei_coordinate_qf=args['sei_coordinate_qf']
    sei_matrix_qf=args['sei_matrix_qf']
    no_analysis_in_decoder = args['no_analysis_in_decoder']

    ### scalable representatin params
    use_scalable_layer = args['scalable_layer_params']['use_scalable_layer']    
    scalable_feature_size_list = args['scalable_layer_params']['feature_size_list']    
    scalable_base_model_name = args['scalable_layer_params']['base_model_name']    
    scalable_feature_code_mode = args['scalable_layer_params']['scalable_feature_code_mode']    
    scalable_feature_diff_mode = args['scalable_layer_params']['scalable_feature_diff_mode']    
    
    

    if 'encoder_type' in args:
        encoder_type = args['encoder_type']
    else:
        encoder_type = model_name

        
    if use_scalable_layer == "ON":    
        
        for qp in qplist:
            for seq in tqdm(seqlist):
                for SRLR_feature_size in scalable_feature_size_list:                    
                    original_seq=Sequence_dir+testingdata_name+'_'+str(seq)+'_'+str(width)+'x'+str(height)+'_25_8bit_444.rgb'                    
                    cmd = "./run_scalable.sh "+model_name+" "+coding_mode+" "+original_seq+" "+str(frames)+" "+str(qp)+" "+str(iframe_format) +" "+ \
                    str(height) +" "+str(width) +" "+str(feature_coding_mode) +" "+str(ae_qf_1) +" "+str(ae_qf_2) +" "+ \
                    str(sei_coordinate_qf) +" "+str(sei_matrix_qf) +" " + str(no_analysis_in_decoder) +" "
                    
                    for blqp in bl_qplist:
                        cmd_blqp = cmd +  use_base_layer + " "+ str(blqp) + " "+ str(SRLR_feature_size)+" "+ str(scalable_base_model_name)+ " " +\
                        str(scalable_feature_code_mode)+ " "+str(scalable_feature_diff_mode)+ " "
                        
                        if coding_mode == 'decoder':
                            cmd_blqp = cmd_blqp + encoder_type
                        os.system(cmd_blqp)                     

                    print(scalable_base_model_name+"_"+model_name+"_"+coding_mode+"_"+testingdata_name+"_"+seq+"_Keyframe QP"+str(qp)+"_SRLR_feature_size_"+str(SRLR_feature_size)+" Finished")





                    
            
