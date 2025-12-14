import os
import math
import torch
import clip
import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

###########10/14 추가 내용. 10/14는 lf-cbm에 remoteclip적용을 위해
import open_clip 

def load_remote_clip(model_name, checkpoint_path, device="cuda"):
    """
    open_clip으로 모델 구조를 만들고, 로컬의 RemoteCLIP 가중치를 씌웁니다.
    retrieval.py의 get_model 함수 로직을 재사용합니다.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained='openai', # 구조만 가져오기 위해 openai pretrain을 사용
        device=device,
        cache_dir='cache/weights/open_clip'
    )
    
    print(f"Loading RemoteCLIP weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 가중치를 덮어씌웁니다.
    msg = model.load_state_dict(checkpoint)
    print("RemoteCLIP weights loaded successfully.")
    print(msg) # <All keys matched successfully> 메시지가 뜨는지 확인
    
    return model, preprocess
###########10/14 추가 내용. 10/14는 lf-cbm에 remoteclip적용을 위해

PM_SUFFIX = {"max":"_max", "avg":""}
#############10/13 추가내용
'''
def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return
'''
# utils.py 파일의 save_target_activations 함수를 아래 코드로 교체하세요.

def save_target_activations(target_model, data, save_names, target_layers, device,
                            target_name, batch_size=128):
    
    # 🌟 [핵심 수정] ViT 모델을 위한 특별 처리 로직 추가
    if 'vit' in target_name:
        print(f"INFO: ViT model detected. Using direct feature extraction (no hooks).")
        all_features = []
        with torch.no_grad():
            for i in tqdm(range(0, len(data), batch_size)):
                image_input = torch.stack([data[j][0] for j in range(i, min(i + batch_size, len(data)))]).to(device)
                features = target_model(image_input)
                all_features.append(features.cpu())
        
        # ViT는 feature_layer가 하나이므로 target_layers[0]을 사용
        save_name = save_names[target_layers[0]]
        print(f"Saving ViT features to {save_name}")
        torch.save(torch.cat(all_features), save_name)
        return # ViT의 경우 여기서 함수 종료

    # --- 이하 코드는 ResNet 등 다른 모델을 위한 기존 로직 ---
    all_features = {}
    for layer in target_layers:
        all_features[layer] = []
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer]))".format(target_layer)
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size)):
            image_input = torch.stack([data[j][0] for j in range(i, min(i + batch_size, len(data)))]).to(device)
            target_model(image_input)
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
#############10/13 추가내용
def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

###########10/14 추가 내용. 10/14는 lf-cbm에 remoteclip적용을 위해

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir, remote_clip_path=None):
    # 10/14 remote_clip_path 인자 추가
    
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    #11/11 토크나이저 추가
    tokenizer = None
    ###########10/14 추가 내용. 10/14는 lf-cbm에 remoteclip적용을 위해
    # 🌟 [핵심 수정] RemoteCLIP 경로 유무에 따라 모델 로딩 방식을 분기합니다.
    if remote_clip_path and os.path.exists(remote_clip_path):
        print("--- Using RemoteCLIP model ---")
        # 1단계에서 추가한 함수를 호출합니다. clip_name은 'ViT-B-32'와 같은 구조 이름입니다.
        clip_model, clip_preprocess = load_remote_clip(clip_name, remote_clip_path, device)
        # 🌟 [수정] open_clip의 토크나이저를 가져옴
        tokenizer = open_clip.get_tokenizer(clip_name)
    else:
        print("--- Using standard OpenAI CLIP model ---")
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
        # 🌟 [수정] 표준 clip의 토크나이저를 가져옴
        tokenizer = clip.tokenize
    ###########10/14 추가 내용. 10/14는 lf-cbm에 remoteclip적용을 위해

    ##### 10/14 삭제 내용 clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    #11/11 수정
    # 1단계에서 보낸 "remote_clip_vit_b_32" 신호를 감지
    elif target_name == 'remote_clip_vit_b_32':
        if not remote_clip_path: # --remote_clip_path가 없으면 에러
            raise ValueError("Backbone 'remote_clip_vit_b_32' requires --remote_clip_path")
        
        print("--- Using RemoteCLIP Model as Target Model (for feature extraction) ---")
        # 🌟 clip_model (RemoteCLIP 전체)을 target_model로 재사용
        target_model = clip_model 
        target_preprocess = clip_preprocess
    #11/11 수정끝
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    #setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    # 🌟 [수정] 하드코딩된 clip.tokenize 대신, 모델에 맞는 'tokenizer' 변수를 사용
    text = tokenizer(["{}".format(word) for word in words]).to(device)
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    # 11/11 수정시작
    if target_name.startswith("clip_") or target_name == 'remote_clip_vit_b_32':
        # 🌟 [수정] remote_clip_vit_b_32 신호도 여기서 처리
        print(f"--- Saving CLIP/RemoteCLIP Image Features for {target_name} ---")
        # 🌟 RemoteCLIP 모델은 encode_image를 사용하므로 이 함수를 호출
        # (주의: target_layers[0]을 사용)
        save_clip_image_features(target_model, data_t, save_names[target_layers[0]], batch_size, device)
    # 11/11 수정끝
    else:
        # 10/13 수정내용
        '''
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
        '''
        save_target_activations(target_model, data_t, save_names, target_layers, device,
                            target_name, batch_size)
        # 10/13 수정내용
    
    return
    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return
########### 10/3 추가 내용
def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=0): # num_workers=0으로 고정
    correct = 0
    total = 0
    
    loader = DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)

            outs, _ = model(images)
            
            # 🌟 [핵심 수정] 출력 텐서의 모양을 확인하고, 필요하면 전치(transpose)합니다.
            if outs.shape[0] != images.shape[0]:
                outs = outs.T # .T는 .transpose(0, 1)과 같습니다.

            pred = torch.argmax(outs, dim=1).cpu()

            correct += torch.sum(pred == labels)
            total += len(labels)
            
    return (correct.item() / total) if total > 0 else 0.0
########### 10/3 추가 내용
def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred
