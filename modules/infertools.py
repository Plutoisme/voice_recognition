import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import toml
import GPUtil
import os
import torch
from modules.FeatCalculator import MelFeatCalculator
from modules.initial_model import initialize_module
import torch.nn.functional as F
import librosa
import numpy as np

os.environ["CUDA_VISIBLE_DEVICEs"] = "1"
class InferEmbedding:
    def __init__(self,
                 configs = None, use_gpu = False):
        '''
        说话人Embedding推理工具
        configs: 模型的配置文件
        use_gpu: 是否使用gpu进行推理
        '''
        assert configs != None, "需要配置文件！"
        # 读入.tar文件， 里面包括model, optimizier, loss等nn.Module, 载入model参数。
        model_path = configs['model']['path']
        model_dict = torch.load(model_path, map_location='cpu')
        model = initialize_module(configs['model']['initial_path'], args = configs['model']['args'])
        model.load_state_dict(model_dict)

        # 获取GPU，确定设备选择
        if use_gpu == True:
            device_ids = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                             excludeID=[], excludeUUID=[])
        self.device = "cuda:"+str(device_ids[0]) if use_gpu == True else 'cpu'
        #print(self.device)
        self.model = model
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def __call__(self, audio):
        if isinstance(audio, str):
            audio, _ = librosa.load(audio, sr=16000)
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0).to(self.device)

        elif isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio)
            audio = audio.unsqueeze(0).to(self.device)

        else:
            raise("推理仅支持音频地址类型以及np.ndarray类型！")
        with torch.no_grad():
            #print(audio[0][5000])  
            featcal = MelFeatCalculator(SpecAug_=False).to(self.device)
            feature = featcal(audio)
            feature1 = feature.reshape(1,-1)
            embedding = self.model(feature)
            embedding = embedding.reshape(1,-1)
            return embedding

# unit test
if __name__ == "__main__":
    
    infer_config_path = '/home/lizhinan/project/voice_recognition/configs/infer.toml'
    configs = toml.load(infer_config_path)
    with torch.no_grad():
        Infer = InferEmbedding(configs=configs,use_gpu=False)
        audio_path = '/home/lizhinan/project/voice_recognition/dataset/cohort_dataset/1.wav'
        embedding = Infer(audio_path)
        embedding = embedding.detach().numpy()
        sum = np.sum(embedding ** 2)
        print(sum)

    








