import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
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
    # RemoteCLIP 로드
    model, preprocess = utils.load_remote_clip(
        args["clip_name"], 
        args["remote_clip_path"], 
        device
    )
else:
    # 일반 CLIP 로드
    import clip
    model, preprocess = clip.load(args["clip_name"], device=device)

model.eval()

# ==========================================
# 3. Grad-CAM 설정
# ==========================================
# 타겟 레이어 설정 (ViT 마지막 블록의 LayerNorm)
target_layers = [model.visual.transformer.resblocks[-1].ln_1]

# GradCAM 객체 생성
cam = GradCAM(model=model.visual, target_layers=target_layers)

# ==========================================
# 4. 이미지 처리 및 실행 (★수정됨★)
# ==========================================
print(f"Processing image: {image_path}")
# 이미지 읽기
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]

# 🌟 [확인] 이미지가 잘 읽혔는지 체크
if rgb_img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# 🌟 [핵심] 강제 리사이징 (64x64 -> 224x224)
rgb_img = cv2.resize(rgb_img, (224, 224))
print(f"Resized image shape: {rgb_img.shape}") # (224, 224, 3)이 나와야 함

# 0~1 정규화 및 텐서 변환
rgb_img_float = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img_float, 
                                mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711]).to(device)

# Grad-CAM 실행
grayscale_cam = cam(input_tensor=input_tensor, targets=None)

# 결과 시각화
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

# 저장
save_path = "gradcam_result.png"
cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"✅ Success! Grad-CAM saved to {save_path}")

# 화면에 보여주기 (Jupyter인 경우)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Result")
plt.axis('off')
plt.show()