import os
import argparse


# 用于将音频文件转化为.wav文件，16k采样率。
parser = argparse.ArgumentParser(description='convert audio to .wav')
parser.add_argument('-I','--input_path',type=str, required=True, help='输入语音的地址')
parser.add_argument('-O','--output_path',type=str, required=True, help='输出.wav文件的地址')
args = parser.parse_args()
try:
    os.system('ffmpeg -i {} -ar 16000 {}'.format(args.input_path, args.output_path))
except:
    print('转码出错，请自行进行文件转换。')
