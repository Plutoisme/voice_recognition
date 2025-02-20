import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import toml
import os
import torch
import torch.nn.functional as F
import numpy as np

from modules.infertools import InferEmbedding

if __name__ == "__main__":
    import argparse
    import torch
    import torch.nn.functional as F
    import toml
    import numpy as np

    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("-C", "--infer_config", 
                       default="/home/lizhinan/project/voice_recognition/configs/infer.toml",
                       help="Configuration (*.toml).")
    parser.add_argument("-S1", "--speaker1", default="/home/lizhinan/project/voice_recognition/dataset/test_dataset/zhangone/a_1.wav")
    parser.add_argument("-S2", "--speaker2", default="/home/lizhinan/project/voice_recognition/dataset/test_dataset/zhangtwo/b_2.wav")
    args = parser.parse_args()

    infer_configs = toml.load(args.infer_config)
    threshold = infer_configs['infer_set']['threshold']

    with torch.no_grad():
        Infer = InferEmbedding(configs=infer_configs, use_gpu=False)
        # 提取声纹特征
        embedding1 = Infer(args.speaker1)
        embedding2 = Infer(args.speaker2)

        # 计算余弦相似度
        similarity = F.cosine_similarity(
            embedding1, embedding2
        ).item()

        # 输出结果
        print("正在将下面两个路径的.wav说话人进行比对:")
        print("说话人1: " + args.speaker1)
        print("说话人2: " + args.speaker2)
        print("------------------------------")
        print(f"声纹相似度得分: {similarity:.4f}")
        print(f"判定阈值: {threshold}")
        
        if similarity >= threshold:
            print("判定结果: 为同一说话人")
        else:
            print("判定结果: 不是同一说话人")
