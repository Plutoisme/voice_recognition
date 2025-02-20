# 从voxceleb1中的语音中，随机抽取N条制作cohortdataset
import os
import glob
import numpy as np
import tqdm
import shutil

vox_path = "/home/lizhinan/project/voice_recognition/dataset/voxceleb1/wav/"
cohort_dataset_path = "/home/lizhinan/project/voice_recognition/dataset/cohort_dataset/"
audiolist = glob.glob(vox_path+"*/*/*.wav")
# cohort_dataset包含N条语音
N = 1000
M = len(audiolist)
indexarray = np.linspace(0,M-1,N)
indexarray = indexarray.astype(np.int32)
os.makedirs(cohort_dataset_path, exist_ok=True)
for i in range(indexarray.shape[0]):
    audio_path = audiolist[i]
    os.system('cp {} {}'.format(audio_path, cohort_dataset_path+str(i)+'.wav'))




