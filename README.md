# 声纹识别
两年前在研究院独立做的课题了，近期整理一下之前的工作，把部分内容开源出来跟大家交流一下。  
该领域最经典的思路是训练一个1vN的神经网络，然后删掉他的最后一层用于分类的全连接层，取其backbone提取语音的embedding作为说话人表征，通过embedding的相似度比对来判断说话人是否同一个人。

### 在线网页Demo
为方便网页展示，相关部分在./deploy文件夹下部署。若需要部署在移动端设备，需要自己训练较小参数量的模型满足实时推理时间需要。经测试可以通过libtorch在安卓端使用JAVA部署，也可以通过NCNN在移动端设备进行部署。

运行./deploy/infer_fastapi.py, 即可在网页实现部署。为方便体验，我将该服务映射在公网IP上，可以通过以下网址进行访问：[http://67.230.171.41:7999](http://67.230.171.41:7999)。PS:(别攻击我，后端菜鸡经不起肉鸡攻击。)
## 项目简介
- 基于9236个说话人(包含中英文)训练的backbone，使用cosine评分方法在**样本较均衡的说话人测试集中**进行比对，**EER为1.11%， minDCF为0.10%**
- 本项目录制了简单的视频介绍，见[]()
- 刚开始做这个项目的时候参考了三个开源项目:
    1. [https://github.com/RookieJunChen/FullSubNet-plus](https://github.com/RookieJunChen/FullSubNet-plus)
    2. [https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch]()
    3. [https://github.com/TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN)
- 支持**ECAPA-TDNN**模型的**预训练**和**finetune**。
- 相较于[https://github.com/TaoRuijie/ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN), 本项目基本完全可复现其在特定数据集上的性能，另外本项目的所训练的模型增添了中文数据，对中文类别下的说话人有更好的**域适应性**。
- 相较于[https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch]()，本项目确定阈值时所使用的数据集更具有**样本均衡性**，所确定的阈值更适用于实际场景。
- 支持finetune(**保证模型可扩展性**)，AS-Norm， Z-Norm等评分方法。

## 快速体验
### 创建环境
```
conda create -n asr python==3.8
```
```
pip install ./requirement.txt
```

### 推理配置及指定说话人路径
#### 同一说话人
```
python ./modules/inferdemo.py -C ./configs/infer.toml -S1 ./dataset/test_dataset/zhangone/a_1.wav -S2 ./dataset/test_dataset/zhangone/a_2.wav

正在将下面两个路径的.wav说话人进行比对:
说话人1: ./dataset/test_dataset/zhangone/a_1.wav
说话人2: ./dataset/test_dataset/zhangone/a_2.wav
------------------------------
声纹相似度得分: 0.8610
判定阈值: 0.43
判定结果: 为同一说话人
```

#### 不同说话人
```
python ./modules/inferdemo.py -C ./configs/infer.toml -S1 ./dataset/test_dataset/zhangone/a_1.wav -S2 ./dataset/test_dataset/zhangtwo/b_2.wav

正在将下面两个路径的.wav说话人进行比对:
说话人1: ./dataset/test_dataset/zhangone/a_1.wav
说话人2: ./dataset/test_dataset/zhangtwo/b_2.wav
------------------------------
声纹相似度得分: 0.0255
判定阈值: 0.43
判定结果: 不是同一说话人
```

## 数据准备
本项目数据训练格式如./configs/train.toml的train_list_zhvoice_and_voxceleb2.txt所示，部分实例如下:
```
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0123.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0124.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0125.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0126.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0127.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0128.wav	0
dataset/zhvoice/zhaishell/S0002/BAC009S0002W0129.wav	0
....................................................    .
dataset/*/*/speaker/audio                               label
```
所有数据被存放在dataset目录下，在train.toml配置文件中指定说话人数量初始化模型。
项目所使用的基座数据主要来自于[voxceleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html)以及[yeyupiaoling所提供的中文数据](https://github.com/yeyupiaoling/VoiceprintRecognition-Pytorch)。

大部分所用说话人数据，以及musan，noise，rir等用于数据增广的数据可以无偿联系作者获取。

## 自定义训练
训练配置见./configs/train.toml，大多数训练设置在其中设置。  

- 你可以自己指定模型所使用的backbone(不仅是ECAPA-TDNN，你可以迁移其他backbone如resnet到此项目中)，本项目公开的backbone完全基于[TaoRuijie的工作](https://github.com/TaoRuijie/ECAPA-TDNN)；

- 你可以尝试不同的损失函数，优化器的组合。
- 关于训练细节，所有训练细节见./modules/trainer/base_trainer，默认所使用的训练器继承于该类，在./modules/trainer/trainer下，你可以拓展自己的训练器。
- 模型训练会在./models下创建本次训练实验目录，下面会记录每个epoch保存的模型参数以及训练日志，部分训练日志如下:
    ```
    [2023-03-17 20:55:11.017052] Train epoch [70/70], batch: [8300/8411], loss: 1.05684, accuracy: 78.69425, lr: 0.00012
    [2023-03-17 20:58:55.016212] Train epoch [70/70], batch: [8400/8411], loss: 1.05699, accuracy: 78.69240, lr: 0.00012
    ```

- 有关模型评估的主要工具见./tools目录，其中eval_network.py用于评估某既定模型的EER，minDCF和最佳1v1所用的阈值。该目录下也包含ASNorm，ZNorm等方法实现(实际部署并不推荐)。
