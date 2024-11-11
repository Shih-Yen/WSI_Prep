#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_features.sh <path_to_patches> <path_to_features>
#
# Notes:
#   I experimented with batch sizes, and 2048 to 3072 seem to work.
#   You need to first create a conda environment and a huggingface
#   token (for UNI) to run this script. Add the token as hf_token
#   argument.
# ==================================================================
#SBATCH -c 2
#SBATCH -t 4:00:00
#SBATCH -p gpu_quad
#SBATCH --mem=32G
#SBATCH -o ./logs/feature_extraction_%A_%a.log
#SBATCH -e ./logs/feature_extraction_%A_%a.log
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# module purge
# module load dgx
# module load miniconda3/24.1.2
# module load cuda12.1/blas/12.1.1

# === CHANGE THESE ===
source activate moe
PATCH_FOLDER=/n/data2/hms/dbmi/kyu/lab/yih796/tile_datasets/TMA_224Stride112_max500_Q0.95_Zoom20X
FEAT_FOLDER=/n/data2/hms/dbmi/kyu/lab/yih796/feature_datasets/TMA_224Stride112_max500_Q0.95_Zoom20X_run2



hf_token="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" # <- add your huggingface token here to use the uni model (https://huggingface.co/settings/tokens)
n_parts=5
IDX=$((SLURM_ARRAY_TASK_ID))
part=$IDX
# n_parts=${3:-1}
# part=${4:-0}

echo "==============================="
nvidia-smi
echo "==============================="

CUDA_VERSION=$(nvidia-smi | awk '/CUDA Version:/ {print $9}')
echo "==============================="
echo "Detected CUDA Version: $CUDA_VERSION"
echo "==============================="

python -c "import torch;print(torch.cuda.is_available())"

export CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"
export CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"


python /n/data2/hms/dbmi/kyu/lab/shl968/MOE/processing/create_features_SY.py \
    --patch_folder "$PATCH_FOLDER" \
    --feat_folder "$FEAT_FOLDER" \
    --slide_type TMA \
    --device cuda \
    --models ctrans,uni,chief,gigapath \
    --hf_token $hf_token \
    --batch_size 2048 \
    --n_parts $n_parts \
    --part $part
