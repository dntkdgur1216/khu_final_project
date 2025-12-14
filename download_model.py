from huggingface_hub import hf_hub_download
import os

# 모델 파일을 저장할 'checkpoints' 폴더 생성
os.makedirs('checkpoints', exist_ok=True)

# 다운로드할 모델 이름 목록
model_names = ['RN50', 'ViT-B-32', 'ViT-L-14']
print("사전 훈련된 RemoteCLIP 모델 다운로드를 시작합니다...")

for model_name in model_names:
    try:
        # huggingface_hub 라이브러리를 통해 모델 자동 다운로드
        checkpoint_path = hf_hub_download(
            repo_id="chendelong/RemoteCLIP",
            filename=f"RemoteCLIP-{model_name}.pt",
            cache_dir='checkpoints'  # 생성한 폴더에 저장
        )
        print(f"✅ {model_name} 모델 다운로드 완료:")
        # 다운로드된 파일의 전체 경로를 출력 (이 경로가 다음 단계에 필요!)
        print(f"   -> {checkpoint_path}")
    except Exception as e:
        print(f"❌ {model_name} 모델 다운로드 중 오류 발생: {e}")