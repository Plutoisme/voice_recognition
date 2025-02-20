import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse, torch, tqdm
import torch.nn.functional as F
from ComputeTools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf, Compute_ASnorm
import toml
from modules.infertools import InferEmbedding


# eval model at the voxceleb1 dataset.
def eval_network(Infer, eval_list, eval_data_path, method = "cosine", cohort_dataset = None):
    files = []
    embeddings = {}
    lines = open(eval_list).read().splitlines()
    for line in lines:
        if os.path.exists(os.path.join(eval_data_path, line.split()[1])):
            files.append(line.split()[1])
        if os.path.exists(os.path.join(eval_data_path, line.split()[2])):
            files.append(line.split()[2])
    setfiles = list(set(files))
    setfiles.sort()
    
    if method == "cosine":
        # 先求得每一个测试音频文件的embedding
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio_path = os.path.join(eval_data_path, file)
            embedding = Infer(audio_path)
            embeddings[file] = [embedding]

        scores, labels = [], []
        # 再统计测试列表中， 每一个音频文件对的相似度得分。
        for line in lines:
            if os.path.exists(os.path.join(eval_data_path, line.split()[1])) and os.path.exists(os.path.join(eval_data_path, line.split()[2])) :
                embedding_enroll = embeddings[line.split()[1]][0]
                embedding_test = embeddings[line.split()[2]][0]
                score = torch.mean(torch.matmul(embedding_enroll, embedding_test.T))
                score = score.detach().cpu().numpy()
                #print(score)
                scores.append(score)
                labels.append(int(line.split()[0]))

    # 如果使用ASnorm统计得分方法， 需要一个冒认语音集合。
    elif method == "asnorm":
        assert cohort_dataset is not None, "asnorm dataset needs cohort dataset!"
        # 先计算冒认语音集合的embeddings, size为[N,embed_dim]。
        cohortDatalist = os.listdir(cohort_dataset)
        for i in range(len(cohortDatalist)):
            assert cohortDatalist[i].endswith('.wav'), "cohortAudio must be .wav file"
            cohortAudioPath = os.path.join(cohort_dataset, cohortDatalist[i])
            cohortAudioEmbedding = Infer(cohortAudioPath)
            if i == 0:
                cohortAudioEmbeddings = cohortAudioEmbedding
            else:
                cohortAudioEmbeddings = torch.cat((cohortAudioEmbeddings,cohortAudioEmbedding), dim=0)
            
        for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
            audio_path = os.path.join(eval_data_path, file)
            embedding = Infer(audio_path)
            embeddings[file] = [embedding]

        scores, labels = [], []
        # 再统计测试列表中， 每一个音频文件对应的相似度得分。 ASnorm方法。
        for line in lines:
            embedding_enroll = embeddings[line.split()[1]][0]
            embedding_test = embeddings[line.split()[2]][0]
            score = torch.mean(torch.matmul(embedding_enroll, embedding_test.T))
            score = Compute_ASnorm(score=score,embedding_enroll=embedding_enroll, embedding_test=embedding_test,
            embedding_cohort = cohortAudioEmbeddings, topk=300)
            scores.append(score.item())
            labels.append(int(line.split()[0]))

    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, min_c_det_threshold = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

    return EER, minDCF, min_c_det_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description= "eval congfig")
    parser.add_argument("--inferConfigPath", default="/home/lizhinan/project/voice_recognition/configs/infer.toml", type=str, help="config_path") # 推理配置文件
    parser.add_argument("--method", type=str, default="cosine", help="the method of calculate the score")
    parser.add_argument("--eval_list", type=str, default="/home/lizhinan/project/voice_recognition/dataset/veri_test2.txt", help="the path of evallist")
    parser.add_argument("--eval_data_path",type=str, default="/home/lizhinan/project/voice_recognition/dataset/voxceleb1/wav", help="where the data for eval in")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    configuration = toml.load(args.inferConfigPath)

    Infer = InferEmbedding(configs = configuration, use_gpu=args.use_gpu)

    EER, minDCF, min_c_det_threshold = eval_network(Infer=Infer,eval_list = args.eval_list, 
                                                    eval_data_path=args.eval_data_path, 
                                                    method = args.method,
                                                    cohort_dataset="/home/lizhinan/project/voice_recognition/dataset/cohort_dataset")

    eval_conclusion_dirname = os.path.dirname(os.path.dirname(configuration['model']['path']))
    eval_conclusion_file = open(os.path.join(eval_conclusion_dirname,"eval_conclusion.txt"),"a+")
    eval_conclusion_file.write("method is {}, EER: {:.2f}%, minDCF: {:.2f}, best threshold: {:.2f} \n".format(args.method,EER,minDCF,min_c_det_threshold))
    eval_conclusion_file.close()






    



        
        

    