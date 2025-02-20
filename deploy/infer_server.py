import _thread
import argparse
import asyncio
import functools
import os
import sys
import time
import toml
import wave
from datetime import datetime
from typing import List
import torch
sys.path.append("/home/lizhinan/project/voice_recognition/")
from modules.infertools import InferEmbedding

#import websockets
from flask import request, Flask, render_template
from flask_cors import CORS

parser = argparse.ArgumentParser(description="Embedding Infer Deploy")
parser.add_argument('--configs', type=str, default='/home/lizhinan/project/voice_recognition/configs/infer.toml', help='配置文件')
parser.add_argument('--host', type=str, default='0.0.0.0', help='监听主机IP')
parser.add_argument('--port_server_enroll', type=int, default=5000, help='注册语音所使用的端口号')
parser.add_argument('--port_server_infer', type=int, default=5001, help='推理embedding进行测试的端口号')
parser.add_argument('--save_path', type=str, default='/home/lizhinan/project/voice_recognition/dataset/enroll_people', help="注册语音的存放地址")
parser.add_argument('--save_test_path', type=str, default='/home/lizhinan/project/voice_recognition/dataset/enroll_people/test_audio', help='测试语音的存放地址' )
parser.add_argument('--use_gpu', type=bool, default=False, help='是否使用GPU')
args = parser.parse_args()

# 设置可访问的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/")
# 允许跨越访问
CORS(app)
configs = toml.load(args.configs)
threshold = configs['infer_set']['threshold']
# 创建推理器
Infer = InferEmbedding(configs = configs, use_gpu = args.use_gpu)
os.makedirs(args.save_test_path, exist_ok=True)


@app.route("/upload_enroll_audio", methods=["POST"])
def upload_enroll_audio():
    f = request.files['audio']
    enroll_person_name = request.form['name']
    if f:
        save_dir = os.path.join(args.save_path, enroll_person_name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{int(time.time()*1000)}{os.path.splitext(f.filename)[-1]}')
        f.save(file_path)
        file_new_path = os.path.splitext(file_path)[0]+'transformed'+'.wav'
        try:
            os.system('ffmpeg -i {} -ar 16000 {}'.format(file_path,file_new_path))
            os.system('rm '+file_path)
        except Exception as e:
            return '文件转换成16k采样率及.wav格式失败，请检查文件格式'
        return "上传注册语音成功，保存在"+enroll_person_name+"文件夹中"
    else:
        return "上传注册语音失败，未接收到语音文件。"



@app.route("/speaker_verification", methods=["POST"])
def speaker_verification():
    f = request.files['audio']
    target_person_name = request.form['name']
    if f:
        file_path = os.path.join(args.save_test_path, f'{int(time.time() * 1000)}{os.path.splitext(f.filename)[-1]}')
        f.save(file_path)
        enroll_audio_dir = os.path.join(args.save_path, target_person_name)
        enroll_audio_list = os.listdir(enroll_audio_dir)
        embedding_test = Infer(file_path)
        scores = []
        logfile = open(os.path.join(args.save_test_path,"log.txt"),'a+')
        start = time.time()
        for i in range(len(enroll_audio_list)):
            enroll_audio_name = enroll_audio_list[i]
            enroll_audio_path = os.path.join(enroll_audio_dir,enroll_audio_name)
            embedding = Infer(enroll_audio_path)
            score = torch.matmul(embedding_test,embedding.T)
            score = score.item()
            logfile.write(file_path+" "+enroll_audio_path+" "+str(score)+'\n')
            scores.append(score)
        end = time.time()
        cost_time = (end-start)/len(enroll_audio_list)
        logfile.close()
        # 计算score平均值与阈值比较：
        sum = 0
        for i in range(len(scores)):
            sum += scores[i]
        if sum/len(scores) >= threshold:
            result = "最后声纹比对得分为{}, 阈值为{}, 平均单次推理耗时{},声纹验证通过".format(sum/len(scores), threshold, cost_time)
        else:
            result = "最后声纹比对得分为{}, 阈值为{}, 平均单次推理耗时{},声纹验证不通过".format(sum/len(scores), threshold, cost_time)
        return result
    else:
        return "没有找到注册人或者测试人的语音信息"

@app.route('/')
def home():
    return render_template("index.html")


# 因为有多个服务需要使用线程启动
def start_server_thread(host, port):
    app.run(host=host, port=port)




if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    _thread.start_new_thread(start_server_thread, (args.host,args.port_server_enroll))
    _thread.start_new_thread(start_server_thread, (args.host,args.port_server_infer))
    # 启动Flask服务
    #server = websockets.serve(stream_server_run, args.host, args.port_stream)
    # 启动WebSocket服务
    #asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()

