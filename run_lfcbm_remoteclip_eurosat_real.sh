#!/bin/bash

#SBATCH --job-name=train_lf-cbm_remoteclip      
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad

#SBATCH -t 1-0                                  
#SBATCH -o logs/lfcbm_remoteclip_eurosat-%A.out
#SBATCH -e logs/lfcbm_remoteclip_eurosat-%A.err

# 11월 중반부터, 실행되는 컴퓨팅 노드를 지정하면 오류가 생기기 시작했다. 그래서 제거함
set -e

echo "Job Started on $(hostname)"
PROJECT_DIR="/data/dntkdgur1216/repos/final_project/Label-free-CBM"
source /data/dntkdgur1216/anaconda3/etc/profile.d/conda.sh
cd $PROJECT_DIR
echo "Changed directory to $(pwd)"
mkdir -p logs saved_activations_remoteclip saved_models_remoteclip data/concept_sets

echo "Running LF-CBM Training with RemoteCLIP..."

# 무슨 버그지? 일단 한줄로 하자
conda run -n lf_remote_env python train_cbm.py --dataset eurosat_rgb --backbone "remote_clip_vit_b_32" --clip_name 'ViT-B-32' --remote_clip_path "../checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt" --concept_set "data/concept_sets/eurosat_concepts2.txt" --activation_dir "saved_activations_remoteclip" --save_dir "saved_models_remoteclip" --device cuda

echo "Job Finished"
