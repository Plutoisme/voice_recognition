import argparse
import os
import random
import sys
import numpy as np
import toml
import torch
from modules.initial_model import initialize_module
from data_utils.dataloader import CustomDataset
from torch.utils.data import DataLoader
import GPUtil

if __name__ == '__main__':
    # 初始化配置
    parser = argparse.ArgumentParser(description="Speaker Verification training on 1 gpu")
    parser.add_argument("-C", "--train_config", default="configs/train.toml",help="Configuration (*.toml).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume the experiment from latest checkpoint.")
    args = parser.parse_args()
    configuration = toml.load(args.train_config)
    device_ids = GPUtil.getAvailable(order = 'first', limit = 8, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_ids[1])

    # 随机种子配置
    torch.manual_seed(configuration['meta']['seed'])
    np.random.seed(configuration["meta"]["seed"])
    random.seed(configuration["meta"]["seed"])

    # 获取数据
    train_dataset = CustomDataset(data_list_path = configuration['meta']['train_list_path'],
                                musan_path = configuration['meta']['musan_path'],
                                rir_path = configuration['meta']['rir_path'],
                                mode = 'train',
                                num_frames = configuration['meta']['num_frames'],
                                aug = True
                                )
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=configuration['meta']['batch_size'],
                              shuffle=True, # sampler已经进行了shuffle, 可以设置为False
                              num_workers=configuration['meta']['num_workers'],
                              pin_memory = configuration['meta']['pin_memory'])
    # 测试数据
    eval_dataset = CustomDataset(data_list_path = configuration['meta']['test_list_path'],
                                musan_path = configuration['meta']['musan_path'],
                                rir_path = configuration['meta']['rir_path'],
                                mode = 'eval',
                                num_frames = configuration['meta']['num_frames'],
                                aug = False
                                )

    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=configuration['meta']['batch_size'],
                             num_workers=configuration['meta']['num_workers'],
                             pin_memory = configuration['meta']['pin_memory'])

    trainer_class = initialize_module(configuration['trainer']['path'],initialize=False)

    trainer = trainer_class(
        config = configuration,
        resume = args.resume,
        train_dataloader = train_loader,
        validation_dataloader = train_loader
    )

    trainer.train()










