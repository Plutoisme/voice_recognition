import torch
import math, torchaudio
import torch.nn as nn
import torch.nn.functional as F
from .BoneModel import ECAPA_TDNN

class SpecAugmentor(nn.Module):

    def __init__(self, freq_mask_width = (0, 8), time_mask_width = (0, 10)):
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width
        super().__init__()

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, fea, time = x.shape
        if dim == 1:
            D = fea
            width_range = self.freq_mask_width
        else:
            D = time
            width_range = self.time_mask_width

        mask_len = torch.randint(width_range[0], width_range[1], (batch, 1), device=x.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, D - mask_len.max()), (batch, 1), device=x.device).unsqueeze(2)
        arange = torch.arange(D, device=x.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange < (mask_pos + mask_len))
        mask = mask.any(dim=1)

        if dim == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)
            
        x = x.masked_fill_(mask, 0.0)
        return x.view(*original_size)

    def forward(self, x):    
        x = self.mask_along_axis(x, dim=2)
        x = self.mask_along_axis(x, dim=1)
        return x

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self,input):
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)
        
class MelFeatCalculator(nn.Module):
    def __init__(self, 
                samplerate = 16000,
                n_fft = 512,
                win_length = 400,
                hop_length = 160,
                f_min = 20,
                f_max = 7600,
                window_fn = torch.hamming_window,
                n_mels = 80,
                SpecAug_ = True,
                prob = 1.0):
        super(MelFeatCalculator,self).__init__()

        self.specaug = SpecAugmentor() if SpecAug_ == True else None
        self.preemphasis = PreEmphasis()

        self.mel_calculator = torchaudio.transforms.MelSpectrogram(
            sample_rate = samplerate,
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            f_min = f_min,
            f_max = f_max,
            window_fn = window_fn,
            n_mels = n_mels
        )
    
    def forward(self, audio):
        with torch.no_grad():
            audio = self.preemphasis.forward(audio) # 预加重
            fea = self.mel_calculator(audio) # 短时傅里叶变换
            LogFea = (fea+1e-6).log() # 取log
            NormLogFea = LogFea - torch.mean(LogFea, dim=-1, keepdim=True) # 在频率这一个维度上进行统计意义上的归一化
            #return NormLogFea
            if self.specaug:
                return self.specaug(NormLogFea) # 频谱增强
            
            else:
                return NormLogFea
import librosa
# Unit Test
if __name__ == "__main__":
    stage = 3
    if stage == 1:
        audio = torch.randn(1,16000)
        audios = torch.randn(12,16000)
        mel_calculator = MelFeatCalculator()
        fea = mel_calculator(audio)
        feas = mel_calculator(audios)
        print(fea.shape)
        print(feas.shape)
    if stage == 2: # for deploying, jit export.
        cal_MelModel = MelFeatCalculator(SpecAug_=False)
        cal_MelModel.to("cpu")
        cal_MelModel.eval()
        scriptModel = torch.jit.script(cal_MelModel)
        torch.jit.save(scriptModel, 'CalMelFeatModel.pth')
    if stage == 3:
        '''
        cal_MelModel = MelFeatCalculator(SpecAug_=False)
        audio, _ = librosa.load("/home/lizhinan/project/voice_recognition/dataset/cohort_dataset/1.wav",16000)
        audio = torch.FloatTensor(audio).unsqueeze(0)
        feature = cal_MelModel(audio)
        model = torch.jit.load("/home/lizhinan/project/voice_recognition/ecapatdnn.pth")
        embed = model(feature)
        embed1 = model(feature)
        print(embed[0][60])
        print(embed1[0][60])
        '''
        model = torch.jit.load("/home/lizhinan/project/voice_recognition/ecapatdnn_script.pth")
        model1 = ECAPA_TDNN(C=1024)
        model1.load_state_dict(torch.load("/home/lizhinan/project/voice_recognition/models/Ecapatdnn_1024_se_zhvoice+vox_0301/checkpoints/model_0070.pth"))
        cal_MelModel = MelFeatCalculator(SpecAug_=False)
        audio, _ = librosa.load("/home/lizhinan/project/voice_recognition/dataset/cohort_dataset/6.wav",16000)
        audio = torch.FloatTensor(audio).unsqueeze(0)
        feature = cal_MelModel(audio)
        model1.eval()
        output = model1(feature)

        output1 = model(feature)
        print(output[0][70])
        print(output1[0][70])

        
