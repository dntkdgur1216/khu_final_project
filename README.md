본 연구에서는 ViT-B/32 아키텍처를 기반으로 한 OpenAI CLIP과 위성 도메인에 특화된 RemoteCLIP 두 가지 모델을 비교 분석하였다.  
데이터셋은 EuroSAT-RGB를 사용하였다.  
또한 본 프로젝트는 Linux GPU 환경 (CUDA 12.1)에서 개발되었다.(KHU 세라프)  

언어: Python 3.9  
핵심 프레임워크: PyTorch 2.4.1 + CUDA 12.1  
모델:  
RemoteCLIP, open-clip-torch 3.2.0 라이브러리를 통해 구현된 ViT-B/32를 불러온 후 해당 논문으로부터 사전 학습된 가중치(Checkpoint)를 로드하여 사용했다.        
OpenAI CLIP, open-clip-torch 3.2.0  

백본모델: ViT-B/32 (Vision Transformer Base-Patch32)    
개념 생성: GPT  
개념 분석: Scikit-learn 1.6.1 (Concept Bottleneck Layer 구현)  
  
생성된 가중치파일은 Label-free-CBM/ evaluate_cbm-clip.ipynb, evaluate_cbm-remoteclip.ipynb에서 성능이 평가된다.  
  
이미지: PIL - Resizing & Loading  
데이터 전처리: Pandas 2.0.3, Numpy 1.26.3  
시각화: Matplotlib, IPython (이미지 렌더링)  


보다 자세한 건 requirements.txt에 있음

사용한 데이터셋 EuroSAT-RGB 주소
https://github.com/phelber/EuroSAT

--------------------------------------
RemoteCLIP 논문의 깃허브 주소
https://github.com/ChenDelong1999/RemoteCLIP

LF-CBM 논문의 깃허브 주소
https://github.com/Trustworthy-ML-Lab/Label-free-CBM

이 두 논문의 코드을 병합한 다음, 전체 코드를 세라프에 옮겨서 실행했다

실행은 run_lfcbm_eurosat.sh (일반 CLIP), run_lfcbm_remoteclip_eurosat.sh (RemoteCLIP) 파일로 실행했고  
하나의 코드에서 일반 CLIP과 RemoteCLIP을 sh파일의 내용에 따라 분기해서 불러올 수 있도록 수정했다.  
(기존 LF-CBM은 일반 CLIP만 불러왔음)  
그게 주된 수정 내용이며 그 후는 자잘한 오류를 수정했다.  
Eurosat-RGB를 불러온다거나, 크기를 늘린다거나 (기본 64*64 이미지).  
정규화가 2번 일어나는걸 막는다거나, 백본모델을 수정하든가, 모델을 가져올떄 사전 훈련된 체크 포인트 값들을 제대로 적용했는가  
train할때 백본 모델을 제대로 freeze를 했는가 등등  
모델의 논리 구조를 수정했다.  
cbm.py, data_utils.py, evaluate_cbm-clip.ipynb, evaluate_cbm-remoteclip.ipynb, similarity.py,  
train_cbm.py, utils.py 가 주된 수정 파일이다.  

이후는 데이터셋을 바꿔가면서 실행했다.  

Label-free-CBM/data/concept_sets/eurosat_concepts2.txt가 사용한 데이터셋이다.  
원래 100개였는데 계속 돌려보면서 필터링해서 제거했다.  
개념 데이터 셋 개수를 늘리면 정확도가 올라가기는 하는데 정작 핵심 근거로 선택한 개념의 질이 낮아져서 개념 개수를 좀 낮게 잡았다.  
일반적인 단어/구 형태말고 UCM_captions 형태를 가진 개념 셋도 만들어서 실험을 해보았는데,    
기본적으로 주어, 동사, 목적어/보어, 수식어를 갖춘 문장 + 전치사를 사용해 객체 간의 위치 관계를 표현 +  
하나의 이미지에 대해 서로 다른 5개의 표현으로 학습되는 특성이 있기에, 이 방법으로는 생성 프롬프트의 개선이 더 필요했다.  
하정확도는 제법 높게 나왔으나, 핵심 근거로 든 개념이 마음에 들지 않아 단어/구 형태로 다시 돌아갔다.  
