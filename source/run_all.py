import sys
import os
import time
import subprocess
import yaml
import warnings
import shutil
warnings.filterwarnings('ignore')

def read_config_file(config_path):
    '''Simply reads a yaml configuration file'''
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def clean_experiment_folders(base_dir):
    '''删除实验目录中的指定文件夹'''
    folders_to_delete = ['dec', 'dec_yuv420', 'enc', 'evaluation', 'evaluation-YUV420', 'kp', 'resultBit']
    
    for folder in folders_to_delete:
        folder_path = os.path.join(base_dir, folder)
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
            except Exception as e:
                pass


class TaskCounter:
    def __init__(self):
        self.total = 0
        self.current = 0
    
    def increment(self):
        self.current += 1
    
    def set_total(self, total):
        self.total = total
    
    def get_progress(self):
        return f"[{self.current}/{self.total}]"


task_counter = TaskCounter()


def run_stage(command, stage_name, model=None):
    global task_counter
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '1'
    env['PYTHONWARNINGS'] = 'ignore'
    if model is not None and model in MODEL_DEVICES:
        command = command + ' --device ' + MODEL_DEVICES[model]
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    elapsed_time = time.time() - start_time
    
    task_counter.increment()
    progress = task_counter.get_progress()
    
    if result.returncode != 0:
        print(f'{progress} [FAIL] {stage_name}')
        raise RuntimeError(f'Stage failed: {stage_name}')
    
    print(f'{progress} [OK] {stage_name} ({elapsed_time:.2f}s)')

############################Please Specify Your Configuration Here########################################

#dataset params 
RGB_SEQ_DIR = '/home/fangzichuan/GFVC_Software/testing_sequences/'  ###You should download the testing sequence and modify the dir.
YUV_SEQ_DIR = '/home/fangzichuan/GFVC_Software/testing_sequences_YUV420/'
DATASETS = ['VOXCELEB'] #
SEQ_LIST = ['001'] #,'002','003','004','005','006','007','008','009','010','011','012','013','014','015']
FEATURE_CODING_MODE = 'AE' 
GOP_SIZE = 25  # GOP size for multi-reference frame coding (100 = single reference frame, i.e., frame 0 only)

ID = '_'+os.path.basename(__file__)[:-3] + '_'.join(DATASETS) + '_'.join(SEQ_LIST) 

#execution params 
PATH_TO_SOURCE = os.path.abspath('.')
PATH_TO_EVAL = os.path.abspath('../evaluate')
CONFIG = 'config_256'

MODELS = ['LVPT'] #,'CFTE','TPSMM','FV2V']
DECODER_MODELS = ['LVPT'] #,'CFTE','TPSMM','FV2V'] 
MODEL_DEVICES = {'LVPT': 'cuda', 'CFTE': 'cuda', 'TPSMM': 'cuda', 'FV2V': 'cuda'}

ENCODE = True
DECODE = True
EVALUATE = True
RGB2YUV = True
EVAL_RGB = True
EVAL_YUV = True
SUMMARY_RESULTS = True
TRANSLATED = False
###Please specify encoder type in decoder config files if you want to execute the translated model.
if TRANSLATED:
    pass

###########################################################################################################

def calculate_total_tasks():
    total = 0
    if ENCODE:
        total += len(DATASETS) * len(MODELS)
    if DECODE:
        total += len(DATASETS) * len(DECODER_MODELS)
    if EVALUATE:
        if RGB2YUV:
            total += len(DATASETS) * len(DECODER_MODELS)
        if EVAL_RGB:
            total += len(DATASETS) * len(DECODER_MODELS)
        if EVAL_YUV:
            total += len(DATASETS) * len(DECODER_MODELS)
    if SUMMARY_RESULTS:
        total += len(DATASETS) * len(MODELS)
    return total

# 读取编码器配置获取 qp_list
encoder_config_path = '../'+CONFIG+'/'+MODELS[0]+'_'+DATASETS[0]+'_enc.yaml'
encoder_config = read_config_file(encoder_config_path)
qp_list = encoder_config.get('qp_list', [])

print('\n' + '='*70)
print('GFVC Pipeline Configuration')
print('='*70)
print(f'Datasets: {DATASETS}')
print(f'Sequences: {SEQ_LIST}')
print(f'Models: {MODELS}')
print(f'Feature Coding Mode: {FEATURE_CODING_MODE}')
print(f'GOP Size: {GOP_SIZE}')
print(f'QP List: {qp_list}')
print(f'Config: {CONFIG}')
print(f'Encode: {ENCODE}, Decode: {DECODE}, Evaluate: {EVALUATE}')
print(f'Total Tasks: {calculate_total_tasks()}')
print('='*70 + '\n')

task_counter.set_total(calculate_total_tasks())
start_time_total = time.time()

# 清理实验文件夹
for model in MODELS:
    config_path = '../'+CONFIG+'/'+model+"_"+DATASETS[0]+"_enc.yaml"
    args = read_config_file(config_path)
    Iframe_format = args['iframe_format'] 
    feature_coding_mode = FEATURE_CODING_MODE
    no_analysis_in_decoder = (args['no_analysis_in_decoder']=='ON')
    if no_analysis_in_decoder:
        assert feature_coding_mode=='AE', 'No analysis in decoder is only supported for AE feature coding mode'
        no_analysis_in_decoder_flag = '+no_analysis_decoding'
    else:
        no_analysis_in_decoder_flag = ''
    width = args['width']
    height = args['height']
    base_dir = '../experiment/'+model+'/Iframe_'+Iframe_format+'/'+ str(width)+'x'+str(height) +'/'+feature_coding_mode+no_analysis_in_decoder_flag+'/'
    clean_experiment_folders(base_dir)

os.chdir(PATH_TO_SOURCE)
dataset_config = {
 'sequence_dir':RGB_SEQ_DIR,
 'yuv_sequence_dir':YUV_SEQ_DIR,
 'seq_list':SEQ_LIST,
 'feature_coding_mode': FEATURE_CODING_MODE,
 'encoder_type': MODELS[0],
 'gop_size': GOP_SIZE
}
dataset_config_path = '../'+CONFIG+'/dataset_config'+ID+'.yaml'
with open(dataset_config_path, "w") as f:
    yaml.dump(dataset_config, f)


if ENCODE:
    print('\n--- ENCODING STAGE ---')
    for dataset in DATASETS:
        for model in MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_enc.yaml"
            run_stage("python run.py --config "+ config_path + ' --dataset_config '+dataset_config_path, f'Encode {model} {dataset}', model)
if DECODE:
    print('\n--- DECODING STAGE ---')
    for dataset in DATASETS:
        for model in DECODER_MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
            decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
            decoder_config = dataset_config.copy()
            decoder_config['encoder_type'] = model
            with open(decoder_dataset_config_path, "w") as f:
                yaml.dump(decoder_config, f)
            run_stage("python run.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path, f'Decode {model} {dataset}', model)
            os.remove(decoder_dataset_config_path)

os.chdir(PATH_TO_EVAL)
if EVALUATE:
    print('\n--- EVALUATION STAGE ---')
    
    if RGB2YUV:
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                run_stage("python rgb444_to_yuv420.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path, f'RGB2YUV {model} {dataset}')
                os.remove(decoder_dataset_config_path)
    if EVAL_RGB:
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                run_stage("python metrics.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path, f'RGB Eval {model} {dataset}')
                os.remove(decoder_dataset_config_path)
    if EVAL_YUV:
        for dataset in DATASETS:
            for model in DECODER_MODELS:
                config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_dec.yaml"
                decoder_dataset_config_path = '../'+CONFIG+'/dataset_config_decoder_'+model+'_'+ID+'.yaml'
                decoder_config = dataset_config.copy()
                decoder_config['encoder_type'] = model
                with open(decoder_dataset_config_path, "w") as f:
                    yaml.dump(decoder_config, f)
                run_stage("python multiMetric_yuv420.py --config "+ config_path + ' --dataset_config '+decoder_dataset_config_path, f'YUV Eval {model} {dataset}')
                os.remove(decoder_dataset_config_path)

if SUMMARY_RESULTS:
    print('\n--- SUMMARY STAGE ---')
    for dataset in DATASETS:
        for model in MODELS:
            config_path = '../'+CONFIG+'/'+model+"_"+dataset+"_enc.yaml"
            run_stage("python bitrate.py --config "+ config_path + ' --dataset_config '+dataset_config_path, f'Bitrate {model} {dataset}')
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

elapsed_time_total = time.time() - start_time_total
print('\n' + '='*70)
print(f'Pipeline Completed Successfully!')
print(f'Total Time: {elapsed_time_total:.2f}s ({elapsed_time_total/60:.2f}m)')
print('='*70 + '\n')


                    
                
