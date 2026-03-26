# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('Agg')

import os, sys
# sys.path.append('/mnt/generative/renjie/GFVC/platform_E2E/GFVC/CFTE_arch_at')
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import OcclusionAwareGenerator, Generator_Test0
from modules.discriminator import MultiScaleDiscriminator
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.RDloss import VideoCompressor ###

import torch
from train import train
from reconstruction import reconstruction
#from animate import animate
import random
import numpy as np

__Generator__ = {
    'base': OcclusionAwareGenerator,
    'test0': Generator_Test0,
}


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_model_by_cfg(cfg, log_dir):
    verbose = False
    #print(cfg)
    with open(cfg) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = os.path.join(log_dir, os.path.basename(cfg).split('.')[0])
        #log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    if verbose:
        print(kp_detector)
 
    videocompressor = VideoCompressor(**config['model_params']['videocompressor_params'])
    
    if torch.cuda.is_available():
        videocompressor.to(opt.device_ids[0])
    if verbose:
        print(videocompressor)
        
    if torch.cuda.is_available():
        videocompressor.to(opt.device_ids[0])
    if verbose:
        print(videocompressor)
    #print(cfg)

    if not os.path.exists(os.path.join(log_dir, os.path.basename(cfg))):
        copy(cfg, log_dir)
    
    # if is_train:
    return generator, discriminator, kp_detector, videocompressor

    
if __name__ == "__main__":
    
    seed_torch(seed=1234)
    
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default="./config/vox-256.yaml")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--arch", default="base", choices=__Generator__.keys())

    parser.add_argument("--log_dir", default='./checkpoint_repro_1946/')
    
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    #parser.add_argument("--checkpoint", default='./checkpoint_repro/', help="path to checkpoint to restore")

    parser.add_argument("--device_ids", default="0,1", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])

    generator = __Generator__[opt.arch](**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    # print(generator)
    # exit()
    # generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
    #                                     **config['model_params']['common_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])

    
    if torch.cuda.is_available():
        discriminator.to(opt.device_ids[0])
    if opt.verbose:
        print(discriminator)

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
    if opt.verbose:
        print(kp_detector)
 
    videocompressor = VideoCompressor(**config['model_params']['videocompressor_params'])
    
    if torch.cuda.is_available():
        videocompressor.to(opt.device_ids[0])
    if opt.verbose:
        print(videocompressor)


    dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)


    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, kp_detector, videocompressor, opt.checkpoint, log_dir, dataset, opt.device_ids)

    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, kp_detector, videocompressor, opt.checkpoint, log_dir, dataset)

    elif opt.mode == 'animate':
        print("Animate...")
        animate(config, generator, kp_detector, videocompressor, opt.checkpoint, log_dir, dataset)