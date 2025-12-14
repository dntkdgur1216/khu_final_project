@echo off
echo ==========================================
echo   Running LF-CBM Training LOCALLY (Windows)
echo ==========================================

:: 1. 작업 폴더로 이동 (현재 파일이 있는 곳)
:: 필요시 cd 명령어로 경로를 명시해도 됩니다.
:: 예: cd /d D:\final_project\repos\final_project\Label-free-CBM

:: 2. Conda 가상환경 활성화
:: 로컬에 'lf_remote_env'라는 환경이 없다면, 현재 사용하는 환경 이름을 쓰세요.
:: 환경이 이미 켜져 있다면 이 줄은 주석(::) 처리해도 됩니다.
call conda activate lf_remote_env

:: 3. 폴더 생성 (없으면 자동 생성)
if not exist "logs" mkdir logs
if not exist "saved_activations_remoteclip" mkdir saved_activations_remoteclip
if not exist "saved_models_remoteclip" mkdir saved_models_remoteclip
if not exist "data\concept_sets" mkdir data\concept_sets

:: 4. RemoteCLIP 가중치 경로 설정 (작성자님이 알려주신 D 드라이브 경로)
:: 파일명 대소문자(PT vs pt)가 윈도우 탐색기에 보이는 것과 똑같은지 꼭 확인하세요!
set REMOTE_CLIP_PATH=D:\final_project\checkpoints\models--chendelong--RemoteCLIP\snapshots\bf1d8a3ccf2ddbf7c875705e46373bfe542bce38\RemoteCLIP-ViT-B-32.pt

:: 5. 학습 실행 명령어
:: 주의: 로컬 GPU 메모리가 부족하면 --batch_size를 512에서 128이나 64로 줄이세요.
echo Starting Training...

python train_cbm.py ^
  --dataset eurosat_rgb ^
  --backbone "remote_clip_vit_b_32" ^
  --clip_name "ViT-B-32" ^
  --remote_clip_path "%REMOTE_CLIP_PATH%" ^
  --concept_set "data/concept_sets/eurosat_concepts.txt" ^
  --activation_dir "saved_activations_remoteclip" ^
  --save_dir "saved_models_remoteclip" ^
  --device cuda ^
  --batch_size 128

echo ==========================================
echo   Job Finished!
echo ==========================================
pause