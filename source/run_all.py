import sys
import os
import time
import yaml 

def read_config_file(config_path):
    '''Simply reads a yaml configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

############################Please Specify Your Configuration Here########################################

#dataset params 
RGB_SEQ_DIR = '/home/fangzichuan/GFVC_Software/testing_sequences/'  ###You should download the testing sequence and modify the dir.
YUV_SEQ_DIR = '/home/fangzichuan/GFVC_Software/testing_sequences_YUV420/'
DATASETS = ['VOXCELEB'] #
SEQ_LIST = ['001'] #,'002','003','004','005','006','007','008','009','010','011','012','013','014','015'] #,'016','017','018','019','020' ] #] #
FEATURE_CODING_MODE = 'SEI' 

ID = '_'+os.path.basename(__file__)[:-3] + '_'.join(DATASETS) + '_'.join(SEQ_LIST) 

#execution params 
PATH_TO_SOURCE = os.path.abspath('.')
PATH_TO_EVAL = os.path.abspath('../evaluate')
CONFIG = 'config_256'

MODELS = ['TPSMM','CFTE','FV2V'] #,'HDAC','MRDAC']#, ,'DAC','HDAC_Extension'] 'FV2V','FOMM',
DECODER_MODELS = ['TPSMM','CFTE','FV2V'] #,'HDAC','MRDAC']#,'DAC','HDAC_Extension','DAC'] 'FV2V',  'FOMM',

ENCODE = True
DECODE = True
EVALUATE = True
RGB2YUV = True
EVAL_RGB = True
EVAL_YUV = True
SUMMARY_RESULTS = True
TRANSLATED = False
###Please specify encoder type in decoder config files if you want to execute the translated model.
if not TRANSLATED:
    DECODER_MODELS = MODELS

###########################################################################################################

os.chdir(PATH_TO_SOURCE)
dataset_config = {
 'sequence_dir':RGB_SEQ_DIR,
 'yuv_sequence_dir':YUV_SEQ_DIR,
 'seq_list':SEQ_LIST,
 'feature_coding_mode': FEATURE_CODING_MODE,
 'encoder_type': MODELS[0]
}
dataset_config_path = '../'+CONFIG+'/dataset_config'+ID+'.yaml'
with open(dataset_config_path, "w") as f:
    yaml.dump(dataset_config, f)


if ENCODE:
    print('Encoding........')
    for dataset in DATASETS:
        for model in MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_enc.yaml"
            os.system("python run.py --config "+ config_path + ' --dataset_config '+dataset_config_path)
if DECODE:
    print('Decoding........')
    for dataset in DATASETS:
        for model in DECODER_MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
            decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
            decoder_config = dataset_config.copy()
            decoder_config['encoder_type'] = model
            with open(decoder_dataset_config_path, "w") as f:
                yaml.dump(decoder_config, f)
            os.system("python run.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path)
            os.remove(decoder_dataset_config_path)

os.chdir(PATH_TO_EVAL)
if EVALUATE:
    print('Evaluating........')
    
    if RGB2YUV:
        print('RGB to YUV converting........')
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                os.system("python rgb444_to_yuv420.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path)
                os.remove(decoder_dataset_config_path)
    if EVAL_RGB:
        print('RGB quality evaluation........')
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                os.system("python metrics.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path)
                os.remove(decoder_dataset_config_path)
    if EVAL_YUV:
        print('YUV quality evaluation........')
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                os.system("python multiMetric_yuv420.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path)
                os.remove(decoder_dataset_config_path)

if SUMMARY_RESULTS:
    print('Summarizing evaluation results')
    for dataset in DATASETS:
        for model in MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_enc.yaml"
            os.system("python bitrate.py --config "+ config_path + ' --dataset_config '+dataset_config_path)
    tim = time.localtime()
    
    
    for i, model in enumerate(MODELS):
        
        for dataset in DATASETS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
            args = read_config_file(config_path)
            Metrics = args['metrics'] 
            Iframe_format=args['iframe_format'] 
            feature_coding_mode = FEATURE_CODING_MODE
            no_analysis_in_decoder = (args['no_analysis_in_decoder']=='ON')
            if no_analysis_in_decoder:
                assert feature_coding_mode=='AE', 'No analysis in decoder is only supported for AE feature coding mode'
                no_analysis_in_decoder_flag = '+no_analysis_decoding'
            else:
                no_analysis_in_decoder_flag = ''
            width = args['width']
            height = args['height']
            if len(args['base_layer_params']['qp']) > 1:
                qp_iter = args['base_layer_params']['qp']
            else:
                qp_iter = args['qp_list'] 
            base_dir = '../experiment/'+model+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/'
            summary_path = os.path.join(base_dir,str(tim[0])+'-'+str(tim[1]).zfill(2)+'-'+str(tim[2]).zfill(2)+'T'+str(tim[3]).zfill(2)+'-'+str(tim[4]).zfill(2)+'-'+str(tim[5]).zfill(2)+'-'+model+'-EvaluationSummary.txt')
            with open(summary_path, 'a') as f:

                f.write('########## '+model+'_'+dataset+'_Bitrate '+'##########'+ '\n')
                bitrate_path = base_dir+'/resultBit/'+dataset+'_resultBit.txt'
                os.makedirs(os.path.dirname(bitrate_path), exist_ok=True)
                with open(bitrate_path, 'r') as bitrate_result:
                    for line in bitrate_result.readlines()[:-1]: 
                        words=line.split()
                        for num in range(len(qp_iter)):
                            f.write(words[num]+ '\n')
                if TRANSLATED:
                    for metric in Metrics:
                        for decoder_model in DECODER_MODELS:
                            if os.path.exists(base_dir+'/evaluation_'+decoder_model+'_dec/'):
                                rgb_result_path=base_dir+'/evaluation_'+decoder_model+'_dec/'+dataset+'_result_'+metric+'.txt'
                                if os.path.exists(rgb_result_path):
                                    f.write('########## '+model+'to'+decoder_model+'_'+dataset+'_RGB444_'+metric+' ##########'+ '\n')
                                    with open(rgb_result_path, 'r') as rgb_result:
                                        for line in rgb_result.readlines()[:-1]: 
                                            words=line.split()
                                            for num in range(len(qp_iter)):
                                                if metric in ['dists','lpips']:
                                                    f.write(str(1-float(words[num]))+ '\n')
                                                else:
                                                    f.write(words[num]+ '\n')
                    for metric in ['psnr','ssim']:
                        for decoder_model in DECODER_MODELS:
                            if os.path.exists(base_dir+'/evaluation_'+decoder_model+'_dec-YUV420/'):
                                yuv_result_path=base_dir+'/evaluation_'+decoder_model+'_dec-YUV420/'+dataset+'_result_'+metric+'.txt'
                                if os.path.exists(yuv_result_path):
                                    f.write('########## '+model+'to'+decoder_model+'_'+dataset+'_YUV420_'+metric+' ##########'+ '\n')
                                    with open(yuv_result_path, 'r') as yuv_result:
                                        for line in yuv_result.readlines()[:-1]: 
                                            words=line.split()
                                            for num in range(len(qp_iter)):
                                                f.write(words[num]+ '\n')
                else:
                    for metric in Metrics:
                        rgb_result_path=base_dir+'/evaluation/'+dataset+'_result_'+metric + '.txt'
                        if os.path.exists(rgb_result_path):
                            f.write('########## '+model+'_'+dataset+'_RGB444_'+metric+' ##########'+ '\n')
                            with open(rgb_result_path, 'r') as rgb_result:
                                for line in rgb_result.readlines()[:-1]: 
                                    words=line.split()
                                    for num in range(len(qp_iter)):
                                        if metric in ['dists','lpips']:
                                            f.write(str(1-float(words[num]))+ '\n')
                                        else:
                                            f.write(words[num]+ '\n')

                    for metric in ['psnr','ssim']:
                        yuv_result_path=base_dir+'/evaluation-YUV420/'+dataset+'_result_'+metric+'.txt'
                        if os.path.exists(yuv_result_path):
                            f.write('########## '+model+'_'+dataset+'_YUV420_'+metric+' ##########'+ '\n')
                            with open(yuv_result_path, 'r') as yuv_result:
                                for line in yuv_result.readlines()[:-1]: 
                                    words=line.split()
                                    for num in range(len(qp_iter)):
                                        f.write(words[num]+ '\n')

os.remove(dataset_config_path)
print('Done!')


                    
                
