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
#SBATCH -t 1:00:00
#SBATCH -p short
#SBATCH --account=yu_ky98
#SBATCH --mem=16G
#SBATCH -o logs/tile_extraction_%j.log
#SBATCH -e logs/tile_extraction_%j.log
#SBATCH --array=0


# === CHANGE THIS ===
module restore fair-tuning-collection
source activate moe

n_parts=100
IDX=$((SLURM_ARRAY_TASK_ID))
part=$IDX
slide_folder="/n/data2/hms/dbmi/kyu/lab/datasets/tcgaCOAD/"
patch_folder="/n/scratch/users/j/joh0984/tcga_tiles/tcgaCOAD"


python create_tiles_bao.py \
    --slide_folder $slide_folder \
    --patch_folder $patch_folder \
    --patch_size 224 \
    --stride 224 \
    --output_size 224 \
    --tissue_threshold 80 \
    --magnifications 40 20 10\
    --n_workers 1 \
    --n_parts $n_parts \
    --part $part \
    --only_coords
