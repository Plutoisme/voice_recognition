import os
import time
import subprocess
import toml
import torch
import librosa
import sys
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
sys.path.append("/home/lizhinan/project/voice_recognition/")
from modules.FeatCalculator import MelFeatCalculator
from modules.initial_model import initialize_module
from fastapi.staticfiles import StaticFiles
import torch.nn.functional as F



# 加载配置文件
configs = toml.load("/home/lizhinan/project/voice_recognition/configs/infer.toml")
threshold = configs['infer_set']['threshold']

# 初始化FastAPI应用
app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 初始化模型
model = initialize_module(configs['model']['initial_path'], args=configs['model']['args'])
model.load_state_dict(torch.load(configs['model']['path'], map_location='cpu'))
model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 全局配置
BASE_SAVE_PATH = "/home/lizhinan/project/voice_recognition/dataset/enroll_people"
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

class AudioProcessor:
    @staticmethod
    def convert_to_16k(input_path: str, output_path: str):
        """转换音频为16kHz采样率"""
        try:
            subprocess.run([
                'ffmpeg',
                '-y',  # 覆盖已存在文件
                '-i', input_path,
                '-ar', '16000',
                '-ac', '1',
                output_path
            ], check=True)
            os.remove(input_path)
            return True
        except Exception as e:
            print(f"音频转换失败: {str(e)}")
            return False

class VoiceEmbeddingGenerator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.feat_calculator = MelFeatCalculator(SpecAug_=False).to(self.device)

    async def generate_embedding(self, audio_path: str):
        """生成音频的声纹嵌入"""
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.feat_calculator(audio_tensor)
                embedding = self.model(features)
                return F.normalize(embedding.reshape(1, -1), p=2, dim=1)
        except Exception as e:
            print(f"生成声纹嵌入失败: {str(e)}")
            raise

embedding_generator = VoiceEmbeddingGenerator()

# 修改静态文件配置
from fastapi.responses import FileResponse

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/enroll")
async def enroll_speaker(
    name: str = Form(...),
    audio: UploadFile = File(...)
):
    """注册说话人接口"""
    try:
        # 创建用户目录
        user_dir = os.path.join(BASE_SAVE_PATH, name)
        os.makedirs(user_dir, exist_ok=True)

        # 保存原始文件
        temp_path = os.path.join(user_dir, f"temp_{int(time.time())}.wav")
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        # 转换音频格式
        final_path = os.path.join(user_dir, f"enroll_{int(time.time())}.wav")
        if not AudioProcessor.convert_to_16k(temp_path, final_path):
            raise HTTPException(status_code=400, detail="音频格式转换失败")

        return JSONResponse(
            content={"message": "注册成功", "path": final_path},
            status_code=200
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_speaker(
    name: str = Form(...),
    audio: UploadFile = File(...)
):
    """验证说话人接口"""
    try:
        # 检查用户是否存在
        user_dir = os.path.join(BASE_SAVE_PATH, name)
        if not os.path.exists(user_dir):
            raise HTTPException(status_code=404, detail="用户未注册")

        # 保存临时验证文件
        temp_dir = os.path.join(BASE_SAVE_PATH, "_temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"verify_{int(time.time())}.wav")
        
        with open(temp_path, "wb") as f:
            f.write(await audio.read())

        # 生成验证音频嵌入
        verify_embedding = await embedding_generator.generate_embedding(temp_path)
        
        # 计算与注册音频的相似度
        scores = []
        for enroll_file in os.listdir(user_dir):
            enroll_path = os.path.join(user_dir, enroll_file)
            enroll_embedding = await embedding_generator.generate_embedding(enroll_path)
            similarity = torch.mm(verify_embedding, enroll_embedding.T).item()
            scores.append(similarity)

        avg_score = sum(scores) / len(scores)
        verification_result = avg_score >= threshold

        return JSONResponse({
            "verification_result": verification_result,
            "similarity_score": avg_score,
            "threshold": threshold,
            "message": "验证通过" if verification_result else "验证失败"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8002,
        log_level="debug"  
    )