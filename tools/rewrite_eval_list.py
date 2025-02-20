import os
from tqdm import tqdm

def remove_error_audio(eval_data_path,eval_list_path):
    lines = open(eval_list_path).read().splitlines()
    new_lines = []
    for line in tqdm(lines):
        if os.path.exists(os.path.join(eval_data_path, line.split()[1])) and os.path.exists(os.path.join(eval_data_path, line.split()[2])) :
            new_lines.append(line)
    with open("/home/lizhinan/project/voice_recognition/test_new_voxceleb1.txt","w+") as f:
        for new_line in new_lines:
            print(new_line)
            f.write(new_line+"\n")
        f.close()

remove_error_audio(eval_data_path="/home/lizhinan/project/dataset/new_voxceleb1/",
eval_list_path="/home/lizhinan/project/dataset/voxceleb1/voxceleb1_H_list.txt")