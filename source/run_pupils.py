# +
import os
import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from GFVC.utils import read_config_file
from GFVC.Pupil.utils.yaml_modify import *
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='../config/pupils.yaml', type=str, help="Path to codec configuration file")
    parser.add_argument('--gfvc_run', default=False, help="Whether to run the gfvc")
    opt = parser.parse_args()
    args = read_config_file(opt.config)

    sequence_dir = args['sequence_dir']
    gfvc_list = args['gfvc']
    iframe_format = args['iframe_format']
    feature_coding_mode = args['feature_coding_mode']
    dataset = args['dataset']
    height = args['height']
    width = args['width']
    qp_list = args['qp_list']
    qf_list = args['pupils_params']['qf_list']
    adaptive_list = args['pupils_params']['adaptive_list']
    if feature_coding_mode == 'AE':
        adaptive_list = ['OFF']
    fps = 25
    # seq_list = {}
    # num_frames = {}
    if 'gfvc_run' in args:
        gfvc_run = args['gfvc_run'] == 'True'
    else:
        gfvc_run = opt.gfvc_run
    if gfvc_run:
        for gfvc in gfvc_list:
            for dset in dataset:
                gfvc_config_path = f'../config/{gfvc}.yaml'
                gfvc_params = {
                    'sequence_dir': args['sequence_dir'],
                    'yuv_sequence_dir': args['yuv_sequence_dir'],
                    'dataset': dset,
                    'iframe_format': args['iframe_format'],
                    'feature_coding_mode': args['feature_coding_mode'],
                    'seq_list': args[f'{dset}_params']['seq_list'],
                    'num_frames': args[f'{dset}_params']['num_frames'],
                    'height': args['height'],
                    'width': args['width'],
                    'qp_list': args['qp_list'],
                    'inference_mode': 'encoder'
                }
                modify_yaml_params(gfvc_config_path, gfvc_params)
                os.system(f'python run.py --config {gfvc_config_path}')

                modify_yaml_key(gfvc_config_path, 'inference_mode', 'decoder')
                os.system(f'python run.py --config {gfvc_config_path}')

    for gfvc in gfvc_list:
        for dset in dataset:
            seq_list = args[f'{dset}_params']['seq_list']
            num_frames = args[f'{dset}_params']['num_frames']
            for qp in qp_list:
                for qf in qf_list:
                    for adaptive in adaptive_list:
                        if adaptive == 'ON':
                            ada_str = 'adaptive'
                        else:
                            ada_str = 'non_adaptive'
                        for seq in seq_list:
                            org_path = sequence_dir + dset + '_' + str(seq) + '_256x256_25fps_8bit_444.rgb'
                            cmd = (
                                f'./run_pupils.sh encoder_pupils {org_path} {gfvc} {num_frames} {qp} {iframe_format} {width} {height}'
                                f' {feature_coding_mode} {qf} {adaptive}')
                            os.system(cmd)

                            cmd = (
                                f'./run_pupils.sh decoder_pupils {org_path} {gfvc} {num_frames} {qp} {iframe_format} {width} {height}'
                                f' {feature_coding_mode} {qf} {adaptive} ')
                            os.system(cmd)








