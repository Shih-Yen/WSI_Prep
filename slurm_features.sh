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
#SBATCH -c 1
#SBATCH -t 0-06:00
#SBATCH -p gpu_dgx
#SBATCH --account=yu_ky98
#SBATCH --mem=32G
#SBATCH -o logs/feature_extraction_%j.out
#SBATCH -e logs/feature_extraction_%j.err
#SBATCH --gres=gpu:1

module purge
module load dgx
module load miniconda3/24.1.2
module load cuda12.1/blas/12.1.1

# === CHANGE THESE ===
source activate /home/che099/.conda/envs/thesis-moe
hf_token="" # <- add your huggingface token here to use the uni model (https://huggingface.co/settings/tokens)

n_parts=${3:-1}
part=${4:-0}

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


python create_features.py \
    --patch_folder $1 \
    --feat_folder $2 \
    --device cuda \
    --models ctrans,lunit,resnet50,uni,swin224,phikon,chief,plip,gigapath,cigar,virchov \
    --hf_token $hf_token \
    --batch_size 2048 \
    --n_parts $n_parts \
    --part $part
