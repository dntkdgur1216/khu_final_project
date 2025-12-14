from huggingface_hub import hf_hub_download
import os

os.makedirs('checkpoints', exist_ok=True)

model_names = ['RN50', 'ViT-B-32', 'ViT-L-14']
print(" pretrain RemoteCLIP 모델 다운로드")

for model_name in model_names:
    try:
        checkpoint_path = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename=f"RemoteCLIP-{model_name}.pt",
            cache_dir='checkpoints'  #
        )
        print(f"{model_name} 모델 다운로드 완료:")
        print(f"   -> {checkpoint_path}")
    except Exception as e:
        print(f" {model_name} 모델 다운로드 중 오류: {e}")
