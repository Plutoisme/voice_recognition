import torch
from torch.cuda.amp import autocast
import tqdm
from .base_trainer import TrainModel
import numpy as np
import time
from datetime import datetime
import os
from ..FeatCalculator import MelFeatCalculator

class Trainer(TrainModel):
    def __init__(self, config, resume,
    train_dataloader, validation_dataloader):
        super(Trainer, self).__init__(config, resume)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader
        self.device = torch.device("cuda")
        self.FeatCalculator = MelFeatCalculator().cuda()


    def _train_epoch(self, epoch, num_epochs):
        self.scheduler.step(epoch-1)
        loss_total = 0.0
        index, top1, loss = 0, 0, 0
        progress_bar = tqdm.tqdm(total=len(self.train_dataloader), desc='Training')
        for batch_id, (audio, label) in enumerate(self.train_dataloader, start=1):
            self.optimizer.zero_grad()
            audio = audio.to(self.device)
            label = label.to(self.device).long()
            with torch.no_grad():
                features = self.FeatCalculator(audio) # 计算特征，features: [batchsize, 80, num_frames]
            speaker_embedding = self.model(features)
            nloss, prec = self.loss_function(speaker_embedding, label)
            # 更新网络参数
            nloss.backward()
            self.optimizer.step()
            # 记录下损失以及准确率
            index += label.shape[0]
            top1 += prec
            loss += nloss.item()
            # 记录训练日志
            progress_bar.update(1)
            progress_bar.refresh()
            if batch_id % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'[{datetime.now()}] '
                    f'Train epoch [{epoch}/{num_epochs}], '
                    f'batch: [{batch_id}/{len(self.train_dataloader)}], '
                    f'loss: {(loss/(batch_id)):.5f}, '
                    f'accuracy: {(top1.item()/index*label.shape[0]):.5f}, '
                    f'lr: {lr:.5f}')
                f_train = open(os.path.join(self.logs_dir,"train_log.txt"), 'a+')
                f_train.write(f'[{datetime.now()}] '
                    f'Train epoch [{epoch}/{num_epochs}], '
                    f'batch: [{batch_id}/{len(self.train_dataloader)}], '
                    f'loss: {(loss/(batch_id)):.5f}, '
                    f'accuracy: {(top1.item()/index*label.shape[0]):.5f}, '
                    f'lr: {lr:.5f}''\n')
                f_train.close()

            
