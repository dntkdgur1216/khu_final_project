# cbm.py 파일 전체를 이 내용으로 덮어쓰세요
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
import data_utils
import utils

class CBM_model(torch.nn.Module):
    def __init__(self, n_concepts, n_classes, backbone, proj_layer, final_layer, proj_mean, proj_std):
        super(CBM_model, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.backbone = backbone
        self.proj_layer = proj_layer
        self.final = final_layer
        self.proj_mean = proj_mean
        self.proj_std = proj_std

    def forward(self, x):
 
        # ... (백본 로직은 그대로) ...
        if hasattr(self.backbone, 'encode_image'):
            x_feat = self.backbone.encode_image(x).float() 
        else:
            x_feat = self.backbone(x) 

        # 1. '원시값' 계산
        proj_c = self.proj_layer(x_feat)

        # 🌟 [수정 2] 훈련 때와 동일하게 '정규화된 값'을 계산합니다.
        # 이 부분이 주석 처리되어 있다면 반드시 해제해야 합니다.
        proj_c_norm = (proj_c - self.proj_mean) / self.proj_std 

        # 🌟 [수정 3] '최종 예측' x가 '원시값'이 아닌 '정규화된 값'을 사용하도록 수정합니다.
        x = proj_c_norm @ self.final.weight.T + self.final.bias

        # 🌟 [수정 4] '해석'을 위해 '정규화된 값'을 반환합니다.
        # plots.py는 이 정규화된 값을 기반으로 기여도를 계산해야 합니다.
        return x, proj_c_norm

def train_glm(image_features, targets, lam, n_iters, lr, n_classes):
    print("Training GLM")
    W_g = torch.randn(n_classes, image_features.shape[1], requires_grad=True, device="cuda")
    b_g = torch.randn(n_classes, requires_grad=True, device="cuda")
    optim = torch.optim.Adam([W_g, b_g], lr=lr)
    for i in tqdm(range(n_iters)):
        logits = image_features@W_g.T + b_g
        loss = F.cross_entropy(logits, targets) + lam*torch.mean(torch.abs(W_g))
        optim.zero_grad()
        loss.backward()
        optim.step()
    return W_g.detach(), b_g.detach()

def learn_projection(image_features, text_features, n_concepts, n_iters=5000, lr=0.001):
    print("Learning projection")
    W_c = torch.randn(n_concepts, image_features.shape[1], requires_grad=True, device="cuda")
    optim = torch.optim.Adam([W_c], lr=lr)
    for i in tqdm(range(n_iters)):
        projected_features = image_features@W_c.T
        norm_proj = projected_features/torch.linalg.norm(projected_features, dim=1, keepdim=True)
        sims = torch.sum(norm_proj*text_features, dim=1)
        loss = -torch.mean(torch.pow(sims, 3))
        optim.zero_grad()
        loss.backward()
        optim.step()
    proj_c = image_features@W_c.T.detach()
    proj_mean = torch.mean(proj_c, dim=0)
    proj_std = torch.std(proj_c, dim=0)
    return W_c.detach(), proj_mean.detach(), proj_std.detach()


def load_cbm(load_dir, device):
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args_dict = json.load(f)
    # 🌟 [수정 시작] 11/11
    backbone_name = args_dict["backbone"]
    remote_clip_path = args_dict.get("remote_clip_path", None) # args.txt에서 path 읽기
    clip_name = args_dict.get("clip_name", "ViT-B-32") # args.txt에서 clip_name 읽기

    if backbone_name == 'remote_clip_vit_b_32':
        if not remote_clip_path:
            raise ValueError("Saved model requires 'remote_clip_path' but it's missing in args.txt")
        print("--- Loading RemoteCLIP Model as Backbone for CBM ---")
        # 🌟 RemoteCLIP 전체 모델을 backbone으로 로드
        backbone, preprocess = utils.load_remote_clip(
            clip_name, 
            remote_clip_path, 
            device
        )
    else:
        # 기존의 일반 timm 모델 로드
        backbone, preprocess = data_utils.get_target_model(backbone_name, device)
    # 🌟 [수정 끝] 11/11
    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)
    n_concepts = W_c.shape[0]
    n_classes = W_g.shape[0]
    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)
    feature_dim = W_c.shape[1]
    
    # 🌟 [수정 3] 랜덤 텐서를 만드는 대신, 불러온 W_c 가중치로 실제 nn.Linear 레이어를 생성합니다.
    proj_layer = torch.nn.Linear(feature_dim, n_concepts, bias=False)
    proj_layer.weight = torch.nn.Parameter(W_c.clone())

    final_layer = torch.nn.Linear(n_concepts, n_classes)
    final_layer.weight = torch.nn.Parameter(W_g.clone())
    final_layer.bias = torch.nn.Parameter(b_g.clone())
    
    model = CBM_model(n_concepts, n_classes, backbone, proj_layer, final_layer, proj_mean, proj_std)
    model.to(device)
    model.eval()
    return model