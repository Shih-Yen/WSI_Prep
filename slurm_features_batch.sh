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

#!/bin/bash
#SBATCH -c 8                          # Request four cores
#SBATCH -t 1:00:00                     # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                     # Partition to run in
##SBATCH --account=yu_ky98_contrib     #
#SBATCH --gres=gpu:1                   # Number of GPUS
#SBATCH --mem=18G                      # Memory total in MiB (for all cores)
#SBATCH -o ./logs/%j_%N_%x.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/%j_%N_%x.out                 # File to which STDERR will be written, including job ID (%j)

#SBATCH -o logs/feature_extraction_%A_%a.log
#SBATCH -e logs/feature_extraction_%A_%a.log
#SBATCH --array=0-1    # <- change this to the number of jobs you want to split the data into


## === PARAMETERS FOR THE FEATURE EXTRACTION ===
## ===         (CHANGE THIS IF NEEDED)       ===
n_parts=2 # <- change this to the number of jobs you want to split the data into
slide_folder="/n/data2/hms/dbmi/kyu/lab/shl968/WSI_for_debug"   # <- change this to the path to your WSI
patch_folder="/n/scratch/users/s/shl968/WSI_prep_test"          # <- change this to the path to your patch coords (from create_tiles.py)
feat_folder="/n/scratch/users/s/shl968/WSI_feat_test"           # <- change this to the path to save your features
STAIN_NORM=false        # <- change this to true if you want to stain normalize your patches
TARGET_MAG=20           # <- change this to the target magnification for your features
## ==============================================

## == LOAD MODULES ==
module restore moe   # O2 modules. Only gcc & cuda is strictly necessary
source activate moe  # install the moe environment using requirements.txt, plus the OpenSlide as described in the README
## ==================



IDX=$((SLURM_ARRAY_TASK_ID))
part=$IDX

export CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"
export CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"

# python /home/bal753/sys_info.py

echo 
nvidia-smi
echo


model=ctrans,chief 
python create_features.py \
 --wsi_folder $slide_folder \
 --patch_folder $patch_folder \
 --feat_folder $feat_folder \
 --n_parts $n_parts \
 --part $part \
 --models $model \
 --target_mag $TARGET_MAG \
 $( [ "$STAIN_NORM" = true ] && echo "--stain_norm" )
