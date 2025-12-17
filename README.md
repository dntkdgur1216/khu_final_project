본 연구에서는 ViT-B/32 아키텍처를 기반으로 한 OpenAI CLIP과 위성 도메인에 특화된 RemoteCLIP 두 가지 모델을 비교 분석하였다.  
데이터셋은 EuroSAT-RGB를 사용하였다.  
또한 본 프로젝트는 Linux GPU 환경 (CUDA 12.1)에서 개발되었다. (KHU 세라프)  

언어: Python 3.9  
핵심 프레임워크: PyTorch 2.4.1 + CUDA 12.1  
모델:  
RemoteCLIP, open-clip-torch 3.2.0 라이브러리를 통해 구현된 ViT-B/32를 불러온 후 RemoteCLIP 논문으로부터 사전 학습된 가중치(Checkpoint)를 로드하여 사용했다.        
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

기존 LF-CBM 코드를 확장하여, 실행 스크립트( run_lfcbm_eurosat.sh (일반 CLIP), run_lfcbm_remoteclip_eurosat.sh (RemoteCLIP) ) 2개로
OpenAI CLIP과 RemoteCLIP을 유연하게 전환하여 로드할 수 있는 동적인 분기 시스템을 구축하였다.
이를 통해 위성 도메인 특화 모델과 일반 모델의 성능을 동일한 환경에서 공정하게 비교·분석할 수 있는 기반을 마련하였다.
(기존 LF-CBM은 일반 CLIP만 불러왔음)  

이후, EuroSAT-RGB의 낮은 원본 해상도(64×64)가 CLIP 기반 백본(ViT-B/32)의 입력 규격과 불일치하는 문제를 해결하기 위해,(224*224)로 업스케일링 했다
데이터 로더와 모델 내부에서 이중으로 정규화가 적용되어 데이터 분포가 왜곡되는 현상을 식별하고, 한번만 정규화를 하도록 오류를 수정하였다.
가중치를 로드할때 사전 훈련된 체크 포인트 값들을 제대로 적용했는지 여러번 확인하였다.
훈련 단계에서 기존 백본 파라미터를 동결하도록 했다.

그외에도 모델의 논리 구조를 수정했으며, 
Label-free-CBM/ cbm.py, data_utils.py, similarity.py, train_cbm.py, utils.py, evaluate_cbm-clip.ipynb, evaluate_cbm-remoteclip.ipynb 가 주된 수정 파일이다.  

이후는 concept set을 바꿔가면서 실행했다.  

Label-free-CBM/data/concept_sets/eurosat_concepts2.txt가 사용한 concept set이다.  
원래는 100개였는데 실험을 진행하면서 필터링했다.

[![Poster Preview](./poster_preview.jpg)](./poster.pdf)
개념 데이터 셋 개수를 늘리면 정확도가 올라가기는 하는데 정작 핵심 근거로 선택한 개념의 질이 낮아져서 개념 개수를 낮게 잡았다.
일반적인 단어/구 형태말고 UCM_captions 형태를 가진 개념 셋도 만들어서 실험을 해보았는데,    
기본적으로 주어, 동사, 목적어/보어, 수식어를 갖춘 문장 + 전치사를 사용해 객체 간의 위치 관계를 표현 +  
하나의 이미지에 대해 서로 다른 5개의 표현으로 학습되는 특성이 있기에, 이 방법으로는 생성 프롬프트의 개선이 더 필요했다.  
정확도는 제법 높게 나왔으나, 핵심 근거로 든 개념이 마음에 들지 않아 단어/구 형태로 다시 돌아갔다.  
