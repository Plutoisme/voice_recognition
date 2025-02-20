import time
from functools import partial
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import toml
import torch
from joblib import Parallel, delayed
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from ..initial_model import initialize_module

def prepare_empty_dir(dirs, resume=False):
    """
    if resume the experiment, assert the dirs exist. If not the resume experiment, set up new dirs.
    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists(), "In resume mode, you must be have an old experiment dir."
        else:
            dir_path.mkdir(parents=True, exist_ok=True)

class TrainModel(nn.Module):
    def __init__(self, config, resume):
        super(TrainModel, self).__init__()
        model = initialize_module(config['model']['path'], args = config['model']['args'])
        loss_function = initialize_module(config['loss_function']['path'], args=config['loss_function']['args'])
        self.model = model.cuda()
        self.loss_function = loss_function.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(),
                                        lr = config['optimizer']['lr'],
                                        weight_decay = config['optimizer']['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.97)

        # Trainer.train in the config
        self.train_config = config["trainer"]["args"]
        self.epochs = config["meta"]['num_epochs']

        # In the 'train.py' file, if the 'resume' item is 'True', we will update the following args:
        self.start_epoch = 1
        self.save_dir = Path(config["meta"]["save_model_dir"]).expanduser().absolute() / config["meta"]["experiment_name"]
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"

        if resume:
            self._resume_checkpoint()

        if config['trainer']['args']['finetune']:
            self._finetune_base_on_backbone(model_path = config['trainer']['args']['pretrain_model_path'], config=config)


        if config["meta"]["preloaded_model_path"]:
            self._preload_model(config["meta"]["preloaded_model_path"])
        # 初始化训练目录
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)
        with open((self.save_dir / f"{time.strftime('%Y-%m-%d %H:%M:%S')}.toml").as_posix(), "w") as handle:
            toml.dump(config, handle)
        self._print_networks([self.model])

    def _finetune_base_on_backbone(self, model_path, config):
        old_model_dict = torch.load(config['trainer']['args']['pretrain_model_path'])
        pretrain_backbone_weight_dict = {k:v for k,v in old_model_dict.items() if k.startswith('backbone.')} # 提取旧模型backbone参数
        model_weight_dict = self.model.state_dict() # 新模型参数
        model_weight_dict.update(pretrain_backbone_weight_dict)# 对新模型参数中backbone部分更新
        self.model.load_state_dict(model_weight_dict)
        self.model.cuda()

        print(f"Model preloaded successfully from {model_path}.")
    
    def _preload_classifierAndloss(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.loss_function.load_state_dict(checkpoint['loss_function'])
        self.loss_function.cuda()
        print(f'Classifier preloaded successfully from {model_path}.')
        

    def _preload_model(self, model_path):
        """
        Preload model parameters (in "*.tar" format) at the start of experiment.

        Args:
            model_path (Path): The file path of the *.tar file
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model'])
        self.model.cuda()
        print(f"Backbone Model preloaded successfully from {model_path}.")

    def _resume_checkpoint(self):
        """
        Resume the experiment from the latest checkpoint.
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."
        # Load it on the CPU and later use .to(device) on the model
        # Maybe slightly slow than use map_location="cuda:<...>"
        # https://stackoverflow.com/questions/61642619/pytorch-distributed-data-parallel-confusion
        checkpoint = torch.load(latest_model_path.as_posix(), map_location="cpu")

        self.start_epoch = checkpoint["epoch"] + 1
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.model.load_state_dict(checkpoint["model"])
        self.loss_function.load_state_dict(checkpoint['loss_function'])
        self.cuda()

        print(f"Model checkpoint loaded. Training will begin at {self.start_epoch} epoch.")

    def _save_checkpoint(self, epoch):
        """
        Save checkpoint to "<save_dir>/<config name>/checkpoints" directory, which consists of:
            - epoch
            - best metric score in historical epochs
            - optimizer parameters
            - model parameters
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")
        state_dict = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        state_dict["model"] = self.model.state_dict()
        state_dict['loss_function'] = self.loss_function.state_dict()
        torch.save(state_dict, (self.checkpoints_dir / f"Model_{str(epoch).zfill(4)}.pth").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix()) # Latest Model.


    @staticmethod
    def _print_networks(models: list):
        print(f"This project contains {len(models)} models, the number of the parameters is: ")

        params_of_all_networks = 0
        for idx, model in enumerate(models, start=1):
            params_of_network = 0
            for param in model.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {idx}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters in the project is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()
        self.loss_function.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()
        self.loss_function.eval()

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("[0 seconds]Begin training...")
            self._set_models_to_train_mode()
            self._train_epoch(epoch, self.epochs)
            self._save_checkpoint(epoch)