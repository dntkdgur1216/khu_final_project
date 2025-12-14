import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# 🌟 [변경] GradCAM 대신 EigenCAM을 사용합니다. ViT에 훨씬 강력합니다.
from pytorch_grad_cam import EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import json
import os

# 사용자 라이브러리
import utils 
import data_utils

# ==========================================
# 1. 설정
# ==========================================
image_path = "data/eurosat/eurosat/2750/Pasture/Pasture_1781.jpg" 
load_dir = "saved_models_remoteclip/eurosat_rgb_cbm_2025_11_23_00_42" 
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 모델 로드
# ==========================================
args_path = os.path.join(load_dir, "args.txt")
with open(args_path, "r") as f:
    args = json.load(f)

print(f"Loading Backbone: {args['backbone']}...")

if args["backbone"] == 'remote_clip_vit_b_32':
    model, preprocess = utils.load_remote_clip(
        args["clip_name"], 
        args["remote_clip_path"], 
        device
    )
else:
    import clip
    model, preprocess = clip.load(args["clip_name"], device=device)

model.eval()

# ==========================================
# 3. Reshape 함수 (ViT 필수)
# ==========================================
def reshape_transform(tensor):
    if tensor.ndim == 3 and tensor.shape[1] == 1:
         tensor = tensor.permute(1, 0, 2)
    
    # CLS 토큰 제거
    patches = tensor[:, 1:, :] 
    patches = patches.permute(0, 2, 1)
    
    batch, dim, _ = patches.shape
    height = width = int(patches.shape[2] ** 0.5) # 7
    
    return patches.reshape(batch, dim, height, width)

# ==========================================
# 4. EigenCAM 설정 (★핵심 수정★)
# ==========================================
# EigenCAM은 마지막 Attention Block 전체를 보는 것이 좋습니다.
target_layers = [model.visual.transformer.resblocks[-1]]

# 🌟 GradCAM -> EigenCAM 교체
cam = EigenCAM(model=model.visual, target_layers=target_layers, reshape_transform=reshape_transform)

# ==========================================
# 5. 이미지 처리 및 실행
# ==========================================
print(f"Processing image: {image_path}")
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]

if rgb_img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 리사이징 (224x224)
rgb_img = cv2.resize(rgb_img, (224, 224))
print(f"Resized image shape: {rgb_img.shape}") 

rgb_img_float = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img_float, 
                                mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711]).to(device)

# 실행
print("Calculating EigenCAM...")
# EigenCAM은 targets가 필요 없습니다.
grayscale_cam = cam(input_tensor=input_tensor)

# 결과 시각화
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 저장
save_path = "EigenCAM_remoteclip_result.png"
cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ Success! Visualization saved to {save_path}")