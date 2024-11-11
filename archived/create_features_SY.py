"""
# Feature Extraction (modified by Shih-Yen Lin)

Creates features based on the patches extracted using create_tiles.py.
The user can select a set of foundational models to extract features from the patches.
Currently, we support the following models:

- lunit
- resnet50
- uni
- swin224
- phikon
- ctrans
- chief
- plip
- gigapath
- cigar



"""

import argparse
import glob
import math
import os
import pprint
import sqlite3
import time
from functools import wraps
from typing import Dict, List, Set

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import openslide

from models.library import get_model, parse_model_type
from utils.transforms import get_transforms
from tqdm import tqdm


# pylint: disable=line-too-long
def parse_args():
    parser = argparse.ArgumentParser(
        description="Create features for patches using pre-defined models.",
        usage="""python create_features_from_patches.py
          --patch_folder PATH_TO_PATCHES
          --feat_folder PATH_TO_features
          [--models MODEL_TYPES]""",
    )
    parser.add_argument(
        "--WSI_list",
        type=str,
        help="List of WSI to process",
        default = '/n/data2/hms/dbmi/kyu/lab/shl968/MOE/TMA_list.csv'
        # default = 'Vienna_GBM_PCNSL_FS(N=97).csv'
    )
    parser.add_argument(
        "--patch_folder",
        type=str,
        help="Root patch folder. Tiled using https://github.com/Shih-Yen/step1_tile_extraction",
        default = '/n/data2/hms/dbmi/kyu/lab/yih796/tile_datasets/TMA_224Stride112_max500_Q0.95_Zoom20X'
        # default='/n/data2/hms/dbmi/kyu/lab/shl968/tile_datasets_for_foundation/Vienna_GBM_PCNSL_FS_224Stride112_max500_Q0.95_Zoom20X'
    )
    parser.add_argument(
        "--feat_folder",
        type=str,
        help="Root folder, under which the features will be stored: <feature_folder>/<slide_id>/",
        default='/n/data2/hms/dbmi/kyu/lab/shl968/feature_datasets/debug'
    )
    parser.add_argument(
        "--slide_type",
        type=str,
        choices=['WSI','TMA'],
        default='TMA',
        help="The type of slide to process (WSI or TMA)"
    )
    
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="The maximum number of tiles to process for each slide. If None, load all tiles(default None)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for processing the patches. Default: 1024.",
    )
    parser.add_argument(
        "--models",
        type=parse_model_type,
        default=[
            "ctrans",
            # "lunit",
            # "resnet50",
            "uni",
            # "swin224",
            # "phikon",
            "chief",
            # "plip",
            "gigapath",
            # "virchov",
            # "cigar",
        ],
        help="Comma-separated list of models to use (e.g., 'lunit,resnet50,uni,swin224,phikon').",
    )
    parser.add_argument(
        "--n_parts",
        type=int,
        default=1,
        help="The number of parts to split the items into (default 1)",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="The part of the total items to process (default 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="The device to use for processing (default 'cuda')",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default="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",
        help="The huggingface token to use for downloading models (default None)",
    )
    return parser.parse_args()


class H5Writer:
    """
    # H5Writer

    Efficient HDF5-based storage for feature extraction results.

    Provides an interface for writing feature tensors and associated metadata to HDF5 files,
    optimized for large-scale feature extraction from image datasets.

    ### Features:
    - Incremental writes with very low memory overhead
    - Automatic dataset creation and extension
    - Gzip compression for storage efficiency
    - Metadata preservation from source files
    - Direct handling of PyTorch tensors

    It supports multiple model outputs in a single file and ensures traceability
    by preserving original metadata.

    ### Usage:
        >>> writer = H5Writer('output_features.h5')
        >>> writer.push_features({'model1': tensor1, 'model2': tensor2})
        >>> writer.copy_metadata('source_data.h5')
        >>> writer.close()

    ### Note:
    This class is not explicitly thread-safe. For multi-threaded environments,
    implement external synchronization mechanisms.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        self.file = h5py.File(self.file_path, "a")

    def push_features(self, features: Dict[str, torch.Tensor]):
        for model_name, feature_tensor in features.items():
            dataset_name = f"{model_name}_features"
            feature_array = feature_tensor.cpu().numpy()

            if dataset_name not in self.file:
                self.file.create_dataset(
                    dataset_name,
                    data=feature_array,
                    maxshape=(None, feature_array.shape[1]),
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )
            else:
                dataset = self.file[dataset_name]
                current_size = dataset.shape[0]
                new_size = current_size + feature_array.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[current_size:new_size] = feature_array

        self.file.flush()
    
    def clear_features(self,model_names:List[str]):
        for model_name in model_names:
            dataset_name = f"{model_name.upper()}_features"
            if dataset_name in self.file:
                del self.file[dataset_name]
        # self.file
        

    def push_metadata_from_SY_csv(self,csv_file:str,max_tiles=None):
        # use for tiles extracted by Shih-Yen's tile extraction script
        # inputs:
        # - csv_file: str, path to the csv file containing the tile information. The csv file should have columns:
        # - max_tiles: int, the maximum number of tiles to load. If None, load all tiles
        # metadata: 
        # - tissue_percentage:
        # - center_x, center_y: the center of the patch
        # - patch_size: the size of the patch
        # - stride: the stride of the patch extraction
        # - output_size: the output size of the patch (usually the same as patch_size)
        df = pd.read_csv(csv_file)
        mag_power = df['mag_level'].values[0]
        patch_size = df['width'].values[0]

        if 'search_mag_level' not in df.columns:
            search_mag_power = 5
        else:
            search_mag_power = df['search_mag_level'].values[0]
        if 'stride_X' not in df.columns:
            stride = patch_size
        else:
            stride = df['stride_X'].values[0]
        if 'output_size' not in df.columns:
            output_size = patch_size
        else:
            output_size = df['output_size'].values[0]
            
        search_patch_size = patch_size // (mag_power/search_mag_power)
        search_patch_area = search_patch_size**2
        for i, row in df.iterrows():
            if max_tiles is not None and i >= max_tiles:
                break
            # write meta data
            if "metadata" not in self.file:
                dtype = np.dtype(
                    [
                        ("tissue_percentage", float),
                        ("center_x", int),
                        ("center_y", int),
                        ("patch_size", int),
                        ("stride", int),
                        ("output_size", int),
                    ]
                )
                self.file.create_dataset(
                    "metadata",
                    (0,),
                    dtype=dtype,
                    maxshape=(None,),
                    chunks=True,
                    compression="gzip",
                    compression_opts=9,
                )
            tissue_percentage = row['nonAllCriteriaSum'] / search_patch_area
            center_x = row['X']+row['width']//2
            center_y = row['Y']+row['height']//2
            patch_size = row['width']

            metadata = self.file["metadata"]
            current_size = metadata.shape[0]
            metadata.resize(current_size + 1, axis=0)
            metadata[current_size] = (
                tissue_percentage,
                center_x,
                center_y,
                patch_size,
                stride,
                output_size,
            )

    
    

    def copy_metadata(self, source_file):
        with h5py.File(source_file, "r") as src:
            if "metadata" in src:
                if "metadata" in self.file:
                    del self.file["metadata"]
                self.file.copy(src["metadata"], "metadata")


    def close(self):
        self.file.close()


def setup_folders(args):
    """Create necessary directories for patches and features if they do not already exist."""
    os.makedirs(args.feat_folder, exist_ok=True)
    assert os.path.exists(
        args.patch_folder
    ), f"Patch folder {args.patch_folder} does not exist."


def load_available_patches(args) -> Set[str]:
    """
    Load the IDs of patches that are available for processing from a text file.

    ## Notes

    This assumes that the tile extraction script still uses the old method with the text file.
    """
    available_patches_txt = f"{args.patch_folder}/success.txt"
    available_patch_ids = set()
    if os.path.exists(available_patches_txt):
        with open(available_patches_txt, "r") as f:
            available_patch_ids = {line.strip() for line in f}
    return available_patch_ids


def retry(max_retries=3, delay=5, exceptions=(Exception,)):
    """Simple decorator to retry a function upon encountering specified exceptions."""

    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = max_retries
            while retries > 0:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries -= 1
                    if retries <= 0:
                        raise
                    print(f"Retry {func.__name__} due to {e}, {retries} retries left.")
                    time.sleep(delay)

        return wrapper_retry

    return decorator_retry



@retry(max_retries=15, delay=5, exceptions=(OSError,))
def load_models(args) -> Dict[str, nn.Module]:
    """
    Load the specified models into memory. Retry a couple of times if they fail.

    ## Why might this fail?

    Huggingface sometimes rate-limits downloads. If we run hundreds of jobs, we may get rate-limited.
    """
    models = {}
    for model in args.models:
        models[str(model)] = get_model(args, str(model)).to(args.device)
    return models


def load_patch(row, patch_folder: str, slide_id: str, transforms: nn.Module):
    """Load an individual patch image and apply transformations for model processing."""
    idx, mag = row["idx"], row["magnification"]
    patch_png = f"{patch_folder}/{slide_id}/{idx}_{mag}.png"
    if os.path.exists(patch_png):
        patch = Image.open(patch_png).convert("RGB")
        return idx, mag, transforms(patch)
    return None, None, None


def get_features(
    batch: torch.Tensor, models: Dict[str, nn.Module]
) -> Dict[str, torch.Tensor]:
    """Process a batch of images using the loaded models and return their features."""
    batch_features = {model_type: [] for model_type in models.keys()}
    with torch.no_grad():
        for model_type, model in models.items():
            batch_features[model_type] = model(batch).detach().cpu()
    return batch_features


def store_metadata(args, slide_id: str):
    """Store patch metadata such as index and magnification to an HDF5 file."""
    metadata = pd.read_csv(f"{args.patch_folder}/{slide_id}.csv")

    # drop column "uuid" if it exists
    if "slide_id" in metadata.columns:
        metadata.drop(columns=["slide_id"], inplace=True)

    for mag in metadata["magnification"].unique():
        filtered_metadata = metadata[metadata["magnification"] == mag]
        with h5py.File(
            f"{args.feat_folder}/{slide_id}/{mag}x_features.h5", "a"
        ) as h5_file:
            if "metadata" in h5_file.keys():
                del h5_file["metadata"]
            h5_file.create_dataset(
                "metadata",
                data=filtered_metadata.to_records(index=False),
                compression="gzip",
            )


def store_features(args, features_dict: Dict[str, torch.Tensor], slide_id: str):
    """Store extracted features into HDF5 files categorized by model and magnification."""
    slide_dir = os.path.join(args.feat_folder, slide_id)
    os.makedirs(slide_dir, exist_ok=True)

    for model_type, mag_features in features_dict.items():
        for mag, features in mag_features.items():
            h5_file_path = os.path.join(slide_dir, f"{mag}x_features.h5")
            with h5py.File(h5_file_path, "a") as h5_file:  # Open in append mode
                model_name = str(model_type).upper()
                features_dataset_name = f"{model_name}_features"
                indices_dataset_name = f"{model_name}_indices"

                features_array = torch.stack(list(features.values())).numpy()
                indices_array = np.array(list(features.keys()), dtype="int")

                # Check if the dataset already exists
                if features_dataset_name in h5_file:
                    # If exists, replace the dataset
                    del h5_file[features_dataset_name]
                if indices_dataset_name in h5_file:
                    del h5_file[indices_dataset_name]

                # Create datasets for features and indices
                h5_file.create_dataset(
                    features_dataset_name,
                    data=features_array,
                    dtype="float32",
                    compression="gzip",
                )
                h5_file.create_dataset(
                    indices_dataset_name,
                    data=indices_array,
                    dtype="int",
                    compression="gzip",
                )



def get_subsample_rate(slide, mag_power=20):
    # Get the subsample rate compared to level 0
    # (necessary since levels in tcgaGBM data is not downsampled by the power of 2)
    
    ## if the file is tiff file, use get_tiff_subsample_rate instead
    if slide._filename.endswith('.tiff') or slide._filename.endswith('.tif'):
        return get_tiff_subsample_rate(slide,mag_power=mag_power)
    
    mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    assert (
        mag >= mag_power
    ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."

    ds_rate = mag / mag_power
    return int(ds_rate)

def get_sampling_params(slide, mag_power=20):
    # Get the optimal openslide level and subsample rate, given a magnification power
    # mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
    # assert (
    #     mag >= mag_power
    # ), f"Magnification of the slide ({mag}X) is smaller than the desired magnification ({mag_power}X)."
    # ds_rate = mag / mag_power
    ds_rate = get_subsample_rate(slide, mag_power=mag_power)

    lvl_ds_rates = np.array(slide.level_downsamples).astype(np.int32)
    levels = np.arange(len(lvl_ds_rates))
    # Get levels that is larger than the given mag power
    idx_larger = np.argwhere(lvl_ds_rates <= ds_rate).flatten()
    lvl_ds_rates = lvl_ds_rates[idx_larger]
    levels = levels[idx_larger]
    # get the closest & larger mag power
    idx = np.argmax(lvl_ds_rates)
    closest_ds_rate = lvl_ds_rates[idx]
    opt_level = levels[idx]
    opt_ds_rate = ds_rate / closest_ds_rate

    return opt_level, opt_ds_rate

def read_region_by_power(slide, start, mag_power, width):
    opt_level, opt_ds_rate = get_sampling_params(slide, mag_power)
    read_width = tuple([int(opt_ds_rate * x) for x in width])
    # start and width should be int
    start = tuple([int(x) for x in start])
    width = tuple([int(x) for x in width])
    im1 = slide.read_region(start, opt_level, read_width)
    if opt_ds_rate != 1:
        im1 = im1.resize(width, resample=Image.LINEAR)
    return im1


def read_TMA(slide, start, mag_power, width):
    # read a region from the TMA image 
    # Inputs:
    # slide: a PIL image
    # start: the start position of the region
    # mag_power: the magnification power (not used in this function)
    # width: the width of the region
    end = [x + offset for x, offset in zip(start,width)]
    # tile = slide[start[0]:end[0],start[1]:end[1],:]
    tile = slide.crop(list(start)+list(end))

    return tile

def load_tiles_from_WSI(slide_csv:pd.DataFrame, slide_path:str,max_tiles=None):
    """
    Load tiles from a WSI
    input:
        slide_csv: str, path to the csv file containing the tile information. The csv file should have columns:
            X,Y,mag_level,width,height,rankVal,rank
        slide_path: str, path to the slide
        max_tiles: int, the maximum number of tiles to load. If None, load all tiles
    """
    df = pd.read_csv(slide_csv)
    # read the slide
    WSI = openslide.OpenSlide(slide_path)
    # read images
    images = []
    for i, row in tqdm(df.iterrows()):
        if max_tiles is not None and i >= max_tiles:
            break
        x,y,mag_level,width,height = row['X'],row['Y'],row['mag_level'],row['width'],row['height']
        start = (x,y)
        width = (width,height)
        im = read_region_by_power(WSI, start, mag_level, width)
        im = im.convert('RGB')
        images.append(im)
    return images


def load_tiles_from_TMA(slide_csv:str, TMA_path:str,max_tiles=None):
    """
    Load tiles from a TMA
    input:
        slide_csv: str, path to the csv file containing the tile information. The csv file should have columns:
            X,Y,mag_level,width,height,rankVal,rank
        slide_path: str, path to the extracted TMA image
        max_tiles: int, the maximum number of tiles to load. If None, load all tiles

    """
    df = pd.read_csv(slide_csv)

    # read the slide
    TMA = Image.open(TMA_path).convert('RGB')
    # read images
    images = []
    for i, row in df.iterrows():
        x,y,mag_level,width,height = row['X'],row['Y'],row['mag_level'],row['width'],row['height']
        start = (x,y)
        width = (width,height)
        # im = read_region_by_power(WSI, start, mag_level, width)
        im = read_TMA(TMA, start=start, mag_power=20, width=width)
        images.append(im)
    return images

    

def process_tiles(args, slide: str, slide_csv:str,slide_id: str=None, models: Dict[str, nn.Module]=None):
    # h5_path = f"{args.patch_folder}/{slide_id}.h5"
    # img_files = glob.glob(f"{slide_folder}/*.{img_ext}")
    if slide_id is None:
        slide_id = os.path.basename(slide)
        
    ## check if csv is empty
    df = pd.read_csv(slide_csv)
    if len(df) == 0:
        print(f"Empty csv file for slide {slide_id}. Skip processing.")
        return
    ## 
    if args.slide_type == 'WSI':
        images = load_tiles_from_WSI(slide_csv, slide,max_tiles=args.max_tiles)
    elif args.slide_type == 'TMA':
        images = load_tiles_from_TMA(slide_csv, slide,max_tiles=args.max_tiles)
    else:
        raise ValueError(f"Unknown slide type: {args.slide_type}")
    
    
    # with h5py.File(h5_path, "r") as h5_file:
    # magnifications = [x for x in h5_file.keys() if "metadata" not in x]
    # total_imgs = sum(len(h5_file[mag]) for mag in magnifications)
    total_imgs = len(images)
    transforms = get_transforms()
    

    with tqdm(total=total_imgs, desc="Processing images") as bar:
        # for mag in magnifications:
        feat_file_path = f"{args.feat_folder}/{slide_id}.h5"
        h5_writer = H5Writer(feat_file_path)
        # if there are existing features, clear them
        h5_writer.clear_features(models.keys())

        mini_batch = []
        h5_writer.push_metadata_from_SY_csv(slide_csv,max_tiles=args.max_tiles)

        for img in images:
            mini_batch.append(transforms(img))
            if len(mini_batch) == args.batch_size:
                mini_batch_tensor = torch.stack(mini_batch).to(args.device)
                features = get_features(mini_batch_tensor, models)
                h5_writer.push_features(features)
                mini_batch = []
                bar.update(args.batch_size)

        # Handle remainder
        if mini_batch:
            mini_batch_tensor = torch.stack(mini_batch).to(args.device)
            features = get_features(mini_batch_tensor, models)
            h5_writer.push_features(features)
            bar.update(len(mini_batch))

        # Copy metadata
        # h5_writer.copy_metadata(h5_path)
        h5_writer.close()

def get_slide_folders(args):
    exclude_folders = ['overlay_vis','overlay_vis_top500','thumbnail','tileStats']
    folders = glob.glob(f"{args.patch_folder}/*")
    folders = [folder for folder in folders if os.path.isdir(folder) and os.path.basename(folder) not in exclude_folders]
    slide_ids = [os.path.basename(folder) for folder in folders]
    return folders, slide_ids

    
def main():
    args = parse_args()
    setup_folders(args)
    # initialize_db(args)

    print("=" * 50)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(vars(args))
    print("=" * 50)
    ## find all available slide folders
    # slide_folders, slide_ids = get_slide_folders(args)
    
    slides_df = pd.read_csv(args.WSI_list,header=None)
    slides = slides_df[0].tolist()
    
    print(f"Found {len(slides)} available slide folders to process.")

    ## split the slides into n_parts
    print(f"Split items into {args.n_parts} parts, processing part {args.part}")
    total_slides = len(slides)
    part_size = math.ceil(total_slides / args.n_parts)
    start_index = args.part * part_size
    end_index = min(start_index + part_size, total_slides)
    ##
    slides = slides[start_index:end_index]
    ##
    print(f"Process slide indices [{start_index}:{end_index}]")

    models = load_models(args)

    # success_data = load_success_data(args)
    requested_experts = set(str(m).upper() for m in args.models)

    for i, slide in enumerate(slides):
        slide_id = os.path.basename(slide)
        print(f"extracting features for slide {i}:\t{slide_id}")   
        slide_csv = glob.glob(os.path.join(f"{args.patch_folder}","tileStats",f"*{slide_id}.csv"))
        assert len(slide_csv) == 1, f"Cannot find the csv file for slide {slide_id}"
        slide_csv = slide_csv[0]
        # assert os.path.exists(slide_csv), f"Cannot find the csv file for slide {slide_id}"
        
        try:
        # args, slide: str, slide_csv:str,slide_id: str=None, models: Dict[str, nn.Module]=None):
            process_tiles(args, slide, slide_csv, slide_id, models)
        except Exception as e:
            print(f"Error processing {slide_id}: {e}")
            continue
        # update_success(args, slide_id, [m.upper() for m in models.keys()])


if __name__ == "__main__":
    main()
