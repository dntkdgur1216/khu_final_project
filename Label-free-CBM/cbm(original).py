import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
import json
import data_utils

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
        x = self.backbone(x)
        # flatten 제거, vit와 호환이 안됨
        # x = torch.flatten(x, 1) 
        x = self.proj_layer(x)
        proj_c = (x-self.proj_mean)/self.proj_std
        
        # 모델 출력이 (배치 크기, 클래스 수)로 되게. 이거 없으면 계속 뒤집힘힘
        x = proj_c @ self.final.weight.T + self.final.bias

        return x, proj_c

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
    backbone, preprocess = data_utils.get_target_model(args_dict["backbone"], device)
    W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)
    n_concepts = W_c.shape[0]
    n_classes = W_g.shape[0]
    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)
    feature_dim = W_c.shape[1]
    
    #  생성한 W_c 가중치를 불러와  nn.Linear 레이어 생성
    proj_layer = torch.nn.Linear(feature_dim, n_concepts, bias=False)
    proj_layer.weight = torch.nn.Parameter(W_c.clone())

    final_layer = torch.nn.Linear(n_concepts, n_classes)
    final_layer.weight = torch.nn.Parameter(W_g.clone())
    final_layer.bias = torch.nn.Parameter(b_g.clone())
    
    model = CBM_model(n_concepts, n_classes, backbone, proj_layer, final_layer, proj_mean, proj_std)
    model.to(device)
    model.eval()
    return model
