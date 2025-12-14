import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
#  GradCAM (CNN) 대신 EigenCAM 사용. ViT 전용이다. 
from pytorch_grad_cam import EigenCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import json
import os

# 사용자 라이브러리
import utils 
import data_utils

image_path = "data/eurosat/eurosat/2750/Pasture/Pasture_1781.jpg" 
load_dir = "saved_models/eurosat_rgb_cbm_2025_11_23_01_23" 
device = "cuda" if torch.cuda.is_available() else "cpu"

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

def reshape_transform(tensor):
    if tensor.ndim == 3 and tensor.shape[1] == 1:
         tensor = tensor.permute(1, 0, 2)
    
    patches = tensor[:, 1:, :] 
    patches = patches.permute(0, 2, 1)
    
    batch, dim, _ = patches.shape
    height = width = int(patches.shape[2] ** 0.5) 
    
    return patches.reshape(batch, dim, height, width)


# EigenCAM 설정 

target_layers = [model.visual.transformer.resblocks[-1]]

# GradCAM -> EigenCAM 
cam = EigenCAM(model=model.visual, target_layers=target_layers, reshape_transform=reshape_transform)

print(f"Processing image: {image_path}")
rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]

if rgb_img is None:
    raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

# EuroSAT 데이터셋들은 64 * 64 인데 EigenCAM은 224 * 224만 받는것같다. / 리사이징 (224x224)
rgb_img = cv2.resize(rgb_img, (224, 224))
print(f"Resized image shape: {rgb_img.shape}") 

rgb_img_float = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img_float, 
                                mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711]).to(device)

# 실행
print("Calculating EigenCAM...")
grayscale_cam = cam(input_tensor=input_tensor)

# 결과 
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

save_path = "EigenCAM_clip_result.png"
cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
print(f"Visualization saved to {save_path}")
