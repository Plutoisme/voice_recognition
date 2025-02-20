import os
import torch
import argparse
import sys
sys.path.append("/home/lizhinan/project/voice_recognition")
from modules.BoneModel import ECAPA_TDNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model export")
    parser.add_argument("-I", "--input_path", default="/home/lizhinan/project/voice_recognition/models/Ecapatdnn_1024_se_zhvoice+vox_0301/checkpoints/model_0070.pth")
    parser.add_argument("-O", "--output_path", default="")
    args = parser.parse_args()



    model = ECAPA_TDNN(C=1024)
    model.load_state_dict(torch.load(args.input_path))
    model = model.to("cpu")
    model = model.eval()
    '''
    input = torch.randn(1,80,200)
    trace_model = torch.jit.trace(model,input)
    torch.jit.save(trace_model, 'ecapatdnn.pth')
    '''

    script_model = torch.jit.script(model)
    torch.jit.save(script_model, 'ecapatdnn_script.pth')
    



    
