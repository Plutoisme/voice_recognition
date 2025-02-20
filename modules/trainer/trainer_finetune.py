import torch
from torch.cuda.amp import autocast
import tqdm
from .base_trainer_ddp import BaseTrainer_DDP
import numpy as np
import time
from datetime import datetime
import os




class Trainer_DDP(BaseTrainer_DDP):
    def __init__(self, dist, rank, config, resume, model, loss_function, optimizer,
    train_dataloader, validation_dataloader):
        super().__init__(dist, rank, config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader
    def _train_epoch(self, epoch, num_epochs):
        loss_total = 0.0
        progress_bar = None
        index, top1, loss = 0, 0, 0

        if epoch == 0: # 当Finetune时，全连接层是新建的，骨架是已有的，我们需要先固定骨架让全连接层先学习一波。
            for para in self.model.module.backbone:
                para.requires_grad = False

        if self.rank == 0:
            progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc='Training')
        for batch_id, (audio, label, _) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            audio = audio.to(self.rank)
            label = label.to(self.rank).long()
            with autocast(enabled=self.use_amp):
                speaker_embedding = self.model.module.backbone(audio) # 采用DDP训练，使用module类调用是必须的。
                #print(speaker_embedding.shape)
                speaker_embedding_ = self.model.module.backbone_end(speaker_embedding)
                #print(speaker_embedding_.shape)
                nloss = self.loss_function.forward(speaker_embedding_, label)[0] # nloss:torch.floattensor, prec:torch.tensor, 
                #print('nloss',nloss)
                #print('prec',prec)
            
            
            # 更新网络参数
            self.scaler.scale(nloss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.clip_grad_norm_ornot == True:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 记录下损失以及准确率
            loss += nloss.detach().cpu().numpy()
            speaker_embedding_ = speaker_embedding_.data.cpu().numpy()
            speaker_embedding_ = np.argmax(speaker_embedding_, axis=1)
            accuracies = []
            label = label.data.cpu().numpy()
            acc = np.mean((speaker_embedding_ == label).astype(int))
            accuracies.append(acc.item())



            # 使用主训练机记录训练日志
            if self.rank==0:
                progress_bar.update(1)
                progress_bar.refresh()
                if batch_id % 100 == 0:
                    print(f'[{datetime.now()}] '
                        f'Train epoch [{epoch}/{num_epochs}], '
                        f'batch: [{batch_id}/{len(self.train_dataloader)}], '
                        f'loss: {(loss/batch_id):.5f}, '
                        f'accuracy: {(sum(accuracies) / len(accuracies)):.5f}, '
                        f'lr: {self.scheduler.get_lr()[0]:.8f}')
                    f_train = open(os.path.join(self.logs_dir,"train_log.txt"), 'a+')
                    f_train.write(f'[{datetime.now()}] '
                        f'Train epoch [{epoch}/{num_epochs}], '
                        f'batch: [{batch_id}/{len(self.train_dataloader)}], '
                        f'loss: {(loss/batch_id):.5f}, '
                        f'accuracy: {(sum(accuracies) / len(accuracies)):.5f}, '
                        f'lr: {self.scheduler.get_lr()[0]:.8f}''\n')
                    f_train.close()

        if epoch == 0: # 当Finetune时，全连接层是新建的，骨架是已有的，我们需要先固定骨架让全连接层先学习一波。
            for para in self.model.module.backbone:
                para.requires_grad = True