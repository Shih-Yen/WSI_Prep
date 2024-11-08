# Processing Utils

Follow this guide to use our processing scripts.

**Basics:** Create a new conda environment and install the requirements.

## Create Tiles

Example:

```bash
python create_tiles.py \
    --slide_folder "/data/wsi" \
    --patch_folder "data/patches/40_20_10" \
    --patch_size 1000 \
    --stride 1000 \
    --output_size 224 \
    --tissue_threshold 5 \
    --magnifications 40 20 10\
    --n_workers 16
```

## Create Features

If you intend to use the `UNI` model, you will need to request access to the model parameters first ([here](https://huggingface.co/MahmoodLab/UNI)). Then add your `hf_token` from [here](https://huggingface.co/settings/tokens) to the `slurm_features.sh` script.

### CTransPath and Chief

Unfortunately, the implementation of ctranspath and chief requires a different `timm` library version than the rest of our code base. Therefore you will have to first download the `timm-0.5.4.tar` file [from here](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view?usp=sharing) into the root of the repo. Then run `install_oldtimm.sh` with your activated environment. This will install the `timm-0.5.4` version as `oldtimm`.

If you are not on O2, you will also need to change the file paths to the model checkpoints in `slurm_features.sh`.

```bash
# example
export CHIEF_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/chief.pth"
export CTRANS_PTH="/n/data2/hms/dbmi/kyu/lab/che099/models/ctranspath.pth"
```

The other models are downloaded on the fly in case you want to use them. Currently we support the following models. We may add more support over time.

- chief
- ctrans
- phikon
- swin224
- resnet50
- lunit
- uni (requires `hf_token`)
- prov-gigapath (requires `hf_token`)
- cigar
- virchov (requires `hf_token`)

For faster processing, it makes sense to split your dataset into parts. You can use a loop to do so quickly.

```bash
n_parts=500
for i in $(seq 0 $((n_parts - 1))); do
    sbatch slurm_features.sh "/home/che099/che099/ebrains/patches" "/home/che099/che099/ebrains/features" $n_parts $i
    sleep 0.1
done
```
