#!/bin/bash

# --- 작업 스케줄러 설정 (SBATCH) ---
#SBATCH --job-name=lfcbm_eurosat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad

#SBATCH -t 1-0
#SBATCH -o logs/lfcbm_eurosat-%A.out
#SBATCH -e logs/lfcbm_eurosat-%A.err

echo "Job Started on $(hostname)"
PROJECT_DIR="/data/dntkdgur1216/repos/final_project/Label-free-CBM"

source /data/dntkdgur1216/anaconda3/etc/profile.d/conda.sh
# 10/ 22 수정, 가상환경을 lf_remote_env로 변경.

# conda activate lfcbm-env
# echo "Activated Conda Env: lfcbm-env"

conda activate lf_remote_env
echo "Activated Conda Env: lf_remote_env"
###### 10/22 수정 끝

cd $PROJECT_DIR
echo "Current Directory: $(pwd)"

mkdir -p logs
mkdir -p saved_models
mkdir -p data/concept_sets

echo "Running LF-CBM pipeline..."

export HF_HOME="/data/dntkdgur1216/huggingface_cache"


# 10/22 수정사항 --activation_dir 'saved_activations' \  추가. 기존 remoteclip의 activation_dir과 구분을 위해.
# 10/22 수정사항 --clip_name 'ViT-B/32' 추가.
python train_cbm.py \
    --dataset 'eurosat_rgb' \
    --backbone 'clip_ViT-B/32' \
    --clip_name 'ViT-B/32' \
    --concept_set 'data/concept_sets/eurosat_concepts4.txt' \
    --activation_dir 'saved_activations' \
    --save_dir 'saved_models' \
    --n_iters 5000 \
    --lam 0.0002 \
    --batch_size 64

echo "Script Finished"
