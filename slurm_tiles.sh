#!/bin/bash
# ==================================================================
# Usage:
#   sbatch slurm_tiles.sh <slide_folder> <patch_folder> [<n_parts> <part>]
#
# Notes:
#   <n_parts> <part> is optional. If you leave it out, it will be 0 and 1.
#   Adjust memory according to your keep_random_n and magnifications
#   arguments.
# ==================================================================
#SBATCH -c 1
#SBATCH -t 0-1:00
#SBATCH -p short
#SBATCH --account=yu_ky98
#SBATCH --mem=1G
#SBATCH -o logs/tile_extraction_%j.log
#SBATCH -e logs/tile_extraction_%j.log

module load gcc/9.2.0
module load cuda/12.1
module load miniconda3/23.1.0

# === CHANGE THIS ===
source activate moe
which python3
python3 --version

n_parts=${3:-0}
part=${4:-1}

python create_tiles.py \
    # --slide_folder $1 \
    # --patch_folder $2 \
    --patch_size 224 \
    --stride 224 \
    --output_size 224 \
    --tissue_threshold 80 \
    --magnifications 40 20 10\
    --n_workers 1 \
    --n_parts $n_parts \
    --part $part
