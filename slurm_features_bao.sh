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
#!/bin/bash
#SBATCH -c 8                          # Request four cores
#SBATCH -t 1-12:00                     # Runtime in D-HH:MM format
#SBATCH -p gpu_quad                     # Partition to run in
##SBATCH --account=yu_ky98_contrib     #
#SBATCH --gres=gpu:1                   # Number of GPUS
#SBATCH --mem=18G                      # Memory total in MiB (for all cores)
#SBATCH --exclude compute-g-17-165
#SBATCH -o ./logs/%j_%N_%x.out                 # File to which STDOUT will be written, including job ID (%j)
#SBATCH -e ./logs/%j_%N_%x.out                 # File to which STDERR will be written, including job ID (%j)

module restore gpu
source activate immune

export CHIEF_PTH="/home/bal753/models/chief.pth"
export CTRANS_PTH="/home/bal753/models/ctranspath.pth"
python /home/bal753/sys_info.py

echo 
nvidia-smi
echo

model=ctrans
python create_features_bao.py --patch_folder /n/scratch/users/b/bal753/DFCI_MOTSU\
 --feat_folder /n/scratch/users/b/bal753/DFCI_MOTSU/$model\
 --wsi_folder /home/bal753/lab/bal753/DFCI\
 --models $model
