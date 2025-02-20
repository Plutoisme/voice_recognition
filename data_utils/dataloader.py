import random
import sys

import warnings
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler
import torch
import os
import librosa
import numpy as np
from torch.utils import data
from scipy import signal
import glob
import random

class CustomDataset(data.Dataset):
    def __init__(self, data_list_path, musan_path, rir_path, num_frames=200, aug = True, mode='train'):
        super(CustomDataset, self).__init__()
        # Dataset初始化
        if data_list_path is not None:
            with open(data_list_path, 'r') as f:
                self.lines = f.readlines()
        self.mode = mode
        self.num_frames = num_frames
        self.aug = aug

        # 配置数据增强
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[1,5], 'music':[1,1]}
        self.noiselist = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

    def __getitem__(self, idx): # if判断过多，待优化改进。
        try:
            audio_path, label = self.lines[idx].replace('\n', '').split('\t')
            audio_path = os.path.join("/home/lizhinan/project/voice_recognition",audio_path)
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            if self.mode == 'train':
                if audio.shape[0] < int(0.5*16000):
                    raise Exception(f"音频长度小于0.5s,不适合训练")
                # 数据增强, 语速处理这一部分要放在这里，不然后面噪音处理会出数组广播bug！！！
                '''
                if self.aug:
                    # 随机改变语速
                    min_speed_rate, max_speed_rate, num_rates, prob = 0.95, 1.1, 5, 0.7
                    rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                    if random.random() < prob:
                        speed_rate = random.choice(rates)
                        old_length = audio.shape[0]
                        new_length = int(old_length / speed_rate)

                        old_indices = np.arange(old_length)
                        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
                        audio = np.interp(new_indices, old_indices, audio)
                '''
                # 语音长度小于训练长度，进行padding
                if audio.shape[0] <= self.num_frames*160 + 240:
                    shortage = self.num_frames*160 + 240 - audio.shape[0]
                    audio = np.pad(audio, (0,shortage), 'wrap')
            
                # 语音长度大于训练长度， 进行随机裁剪
                else:
                    start = random.randint(0, audio.shape[0] - (self.num_frames*160 + 240) - 1)
                    stop = start + self.num_frames*160 + 240
                    audio = audio[start:stop]
                
                # 最后获取一个
                
                # 数据增强
                if self.aug:
                    # 随机改变音量
                    min_gain_dBFS, max_gain_dBFS, prob_1 = -5, 5, 0.7
                    if random.random() < prob_1:
                        gain = random.uniform(min_gain_dBFS, max_gain_dBFS)
                        audio = audio* 10**(gain/20)

                    # 随机增加噪声品种
                    augtype = random.randint(0,5)
                    #augtype = 1
                    if augtype == 0:
                        audio = audio
                    if augtype == 1: # 增加混响
                        audio = self.add_rir(audio)
                    if augtype == 2: # 增加其他人说话干扰
                        audio = self.add_noise(audio, 'speech')
                    if augtype == 3: # 增加音乐
                        audio = self.add_noise(audio, 'music')
                    if augtype == 4: # 增加噪声
                        audio = self.add_noise(audio, 'noise')
                    if augtype == 5: # 增加多元干扰
                        audio = self.add_noise(audio, 'speech')
                        audio = self.add_noise(audio, 'music')

            if self.mode == "eval": 
                # padding      
                if audio.shape[0] <= self.num_frames*160 + 240:
                    shortage = self.num_frames*160 + 240 - audio.shape[0]
                    audio = np.pad(audio, (0,shortage), 'wrap')
                # 进行固定裁剪
                else:
                    audio = audio[:self.num_frames*160 + 240]

            if np.max(audio) <= 1:
                return torch.FloatTensor(audio), np.array(int(label), dtype=np.int64)
            else: # 防止加噪后数值爆表，做一个归一化处理。
                return torch.FloatTensor(audio / np.max(audio)), np.array(int(label), dtype=np.int64)
        # 异常检测
        except Exception as ex:
            print(f"[{datetime.now()}] 数据: {self.lines[idx]} 出错，错误信息: {ex}", file=sys.stderr)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.lines)
    
    def add_rir(self, audio):
        rir_file = random.choice(self.rir_files)
        rir, sr = librosa.load(rir_file, sr=16000)
        rir = np.expand_dims(rir,0)
        # 扩展维度后做卷积，实现房间脉冲响应。
        rir = rir / np.sqrt(np.sum(rir**2))
        audio = np.expand_dims(audio,0)
        enhanced_audio = signal.convolve(audio, rir, mode="full")[:,:self.num_frames*160 + 240]

        return enhanced_audio[0]

    def add_noise(self, audio, noise_mode):
        # 本项目读取音频采用librosa库，其底层使用soundfile实现。
        # sounfile计数区间：[-32768,32767], librosa计数区间[-1,1]
        # 分贝计算公式： db = 10 * log10( mean(audio**2)/ref ), 这里的ref = 32768^2 假设音频采用PCM16位量化
        # 因此 若用librosa读取， db = 10 * log10(mean(audio**2)+1e-9) +20 * log10(32768)
        clean_db = 10 * np.log10(np.mean(audio**2)+1e-9) + 20*np.log10(32768)
        numnoise = self.numnoise[noise_mode]
        noiselist = random.sample(self.noiselist[noise_mode], random.randint(numnoise[0], numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)
            length = self.num_frames*160 + 240
            # 如果噪声比语音短，对噪声进行补零
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = np.pad(noiseaudio, (0, shortage), 'wrap')
            # 如果噪声比语音长， 对噪声进行截取
            else:
                start = random.randint(0, noiseaudio.shape[0] - length - 1)
                stop = start + length
                noiseaudio = noiseaudio[start:stop]

            noiseaudio = np.stack([noiseaudio],0)
            noise_db = 10 * np.log10(np.mean(noiseaudio**2)+1e-9) + 20*np.log10(32768)
            noisesnr = random.uniform(self.noisesnr[noise_mode][0], self.noisesnr[noise_mode][1])
            noises.append(np.sqrt(10**((clean_db - noise_db - noisesnr) / 10))*noiseaudio)
        noise = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True)
        audio = np.stack([audio],axis=0)
        enhanced_audio = noise + audio

        return enhanced_audio[0]



# Unit Test
import toml
import soundfile
if __name__ == "__main__":
    data_list_path = "/home/lizhinan/project/voice_recognition/dataset/train_list_zhvoice.txt"
    configuration = toml.load("/home/lizhinan/project/voice_recognition/configs/train.toml")
    train_dataset = CustomDataset(configuration['meta']['train_list_path'],
                                configuration['meta']['musan_path'],
                                "/home/lizhinan/project/voice_recognition/dataset/RIRS_NOISES/simulated_rirs",
                                mode='train',
                                num_frames=configuration['meta']['num_frames'],
                                aug=True)
    '''
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=configuration['meta']['batch_size'],
                              shuffle=True, # sampler已经进行了shuffle
                              num_workers=configuration['meta']['num_workers'],
                              pin_memory = configuration['meta']['pin_memory']) 
    for batch_id, (audio, label) in enumerate(train_loader):
        print(batch_id,audio,label)
    '''
    #print(train_dataset.noiselist)

    dirs = "/home/lizhinan/project/voice_recognition/dataset/train_audios_example"
    os.makedirs(dirs, exist_ok=True)
    for i in range(2):
        soundfile.write(os.path.join(dirs,str(i)+'.wav') ,train_dataset[i][0].detach().cpu().numpy(), samplerate=16000)
        print(train_dataset[i][0].shape)



    
     





