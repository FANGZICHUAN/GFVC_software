import matplotlib

matplotlib.use('Agg')

import os, sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime
from shutil import copy

from mydataset import myDataset

from modules.generator import OcclusionAwareGenerator
from modules.discriminator import MultiScaleDiscriminator


import torch

from train import train
from reconstruction import reconstruction
# from animate import animate
import warnings

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    warnings.filterwarnings('ignore')

    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "animate"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.add_argument("--device", default="cpu", help="device to train")
    parser.add_argument("--eye", default="right", help="eye to train")
    parser.add_argument("--datasetID", default=0, help="dataset to train")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    # 加载配置文件
    with open(opt.config) as f:
        config = yaml.safe_load(f)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += '_' + opt.eye + '_' + strftime("%d_%m_%y_%H.%M.%S", gmtime())

    if opt.device == "gpu" or opt.device == "cuda":
        my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        my_device = torch.device("cpu")

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    generator = generator.to(my_device)
    # **关键字参数、*位置参数

    # 检查CUDA是否可用，并将生成器模型移动到指定的设备上（如果可用）。如果设置了verbose标志，则打印生成器模型的结构。
    # if torch.cuda.is_available():
    #     generator.to(opt.device_ids[0])
    # if opt.verbose:
    #     print(generator)

    discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                            **config['model_params']['common_params'])
    discriminator = discriminator.to(my_device)
    # if torch.cuda.is_available():
    #     discriminator.to(opt.device_ids[0])
    # if opt.verbose:
    #     print(discriminator)

    dataset = myDataset(is_train=(opt.mode == 'train'), eye=opt.eye, datasetID=opt.datasetID, **config['dataset_params'])  # 重点研究一下

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, discriminator, opt.checkpoint, log_dir, dataset, my_device, opt.eye)
    elif opt.mode == 'reconstruction':
        print("Reconstruction...")
        reconstruction(config, generator, opt.checkpoint, log_dir, dataset, my_device, opt.eye, show=1)
    # elif opt.mode == 'animate':
    #     print("Animate...")
    #     animate(config, generator, kp_detector, opt.checkpoint, log_dir, dataset)
