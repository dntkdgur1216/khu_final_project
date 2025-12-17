본 연구에서는 ViT-B/32 아키텍처를 기반으로 한 OpenAI CLIP과 위성 도메인에 특화된 RemoteCLIP  
두 가지 모델의 성능과 설명 가능성을 비교 분석하는 것을 목표로 합니다.  
이를 위해 Label-Free Concept Bottleneck Model을 적용하였으며, 각 모델이 위성 이미지를 분류할 때  
어떤 개념(Concept)을 핵심 근거로 삼는지 시각화하고 분석하였습니다.  
데이터셋은 EuroSAT-RGB를 사용하였습니다.  

사용한 데이터셋 EuroSAT-RGB 주소  
https://github.com/phelber/EuroSAT  

RemoteCLIP 논문의 깃허브 주소  
https://github.com/ChenDelong1999/RemoteCLIP  

LF-CBM 논문의 깃허브 주소  
https://github.com/Trustworthy-ML-Lab/Label-free-CBM  

본 프로젝트는 Linux GPU 서버 (KHU Seraph) 환경에서 개발 및 테스트 되었습니다. 상세 패키지 정보는 requirements.txt를 참고해 주세요.  

언어: Python 3.9  
핵심 프레임워크: PyTorch 2.4.1 + CUDA 12.1  
모델 / 정답 메트릭스 생성시 사용 :  
RemoteCLIP: 논문에서 제공된 사전 학습된 가중치를 로드하여 사용하였습니다. 이미지 인코더와 텍스트 인코더 모두 사용하였습니다.  
(RemoteCLIP-VIT-B-32.pt)  
OpenAI CLIP: open-clip-torch 3.2.0 라이브러리를 통해 OpenAI 공식 사전 학습 가중치를 로드하여 사용하였습니다.

백본모델: ViT-B/32 (Vision Transformer Base-Patch32)  / clip_ViT-B/32과 remote_clip_vit_b_32로 구분됩니다.  
clip_ViT-B/32: open-clip-torch를 통해 Standard OpenAI CLIP의 가중치를 그대로 사용하였습니다.  
remote_clip_vit_b_32: open-clip-torch로 ViT-B/32 구조를 생성한 후,   
사전 학습된 가중치 (RemoteCLIP-VIT-B-32.pt)를 로드하여 이미지 인코더 가중치를 교체하였습니다.  

개념 생성: GPT-4 (Concept Set 생성)  
개념 분석: Scikit-learn 1.6.1 (Concept Bottleneck Layer 구현)  
  
생성된 가중치파일은 Label-free-CBM/ evaluate_cbm-clip.ipynb, evaluate_cbm-remoteclip.ipynb에서 성능이 평가됩니다.  
  
이미지: PIL - Resizing & Loading  
데이터 전처리: Pandas 2.0.3, Numpy 1.26.3  
시각화: Matplotlib, IPython (이미지 렌더링)   

기존 LF-CBM과 RemoteCLIP의 코드를 통합하고, 실행 스크립트( run_lfcbm_eurosat.sh (일반 CLIP), run_lfcbm_remoteclip_eurosat.sh (RemoteCLIP) ) 2개로  
OpenAI CLIP과 RemoteCLIP을 유연하게 전환하여 로드할 수 있는 동적인 분기 시스템을 구축하였습니다.  
이를 통해 위성 도메인 특화 모델과 일반 모델의 성능을 동일한 환경에서 공정하게 비교·분석할 수 있는 기반을 마련하였습니다.  
(기존 LF-CBM은 일반 CLIP만 불러왔음)   

이후, EuroSAT-RGB의 낮은 원본 해상도(64×64)가 CLIP 기반 백본(ViT-B/32)의 입력 규격과 불일치하는 문제를 해결하기 위해,(224*224)로 업스케일링 했습니다.  
데이터 로더와 모델 내부에서 이중으로 정규화가 적용되어 데이터 분포가 왜곡되는 현상을 식별하고, 한번만 정규화를 하도록 오류를 수정하였습니다.  
가중치를 로드할때 사전 훈련된 체크 포인트 값들을 제대로 적용했는지 여러번 확인하였습니다.  
훈련 단계에서 기존 백본 파라미터를 동결하도록 했습니다.  

그외에도 모델의 논리 구조를 수정했으며,  
Label-free-CBM/ cbm.py, data_utils.py, similarity.py, train_cbm.py, utils.py, evaluate_cbm-clip.ipynb, evaluate_cbm-remoteclip.ipynb 가 주된 수정 파일입니다.  

이후는 concept set을 최적화하는 과정을 거쳤습니다.  

초기 100개의 개념에서 시작하여 필터링 과정을 거쳐 최종적으로  
Label-free-CBM/data/concept_sets/eurosat_concepts.txt를 확정하였습니다.  

개념의 개수를 무작정 늘리면 분류 정확도는 상승하지만, 설명력 측면에서 핵심 근거의 질이 하락하는 트레이드오프를 확인하였습니다.  
따라서 가장 직관적인 설명력을 제공하는 최적의 개수로 설정하였습니다.  
일반적인 단어/구 형태말고 위성 캡션 표준인 UCM_captions 스타일의 문장형 개념 셋도 만들어서 실험을 해보았지만( RemoteCLIP은 UCM_captions 스타일의 문장형 텍스트로 학습되었습니다.)  
기본적으로 주어, 동사, 목적어/보어, 수식어를 갖춘 문장 + 전치사를 사용해 객체 간의 위치 관계를 표현 +  
하나의 이미지에 대해 서로 다른 5개의 표현으로 학습되는 특성이 있기에, 현재로서는 생성 프롬프트의 개선이 더 필요합니다.
정확도는 제법 높게 나왔으나, 핵심 근거로 든 개념이 마음에 들지 않아 단어/구 형태로 다시 돌아갔습니다.

![Project Poster](./poster.jpg)  
