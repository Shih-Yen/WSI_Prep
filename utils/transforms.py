import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


def get_transforms(train=False):
    """
    Takes a list of images and applies the same augmentations to all of them.
    This is completely overengineered but it makes it easier to use in our pipeline
    as drop-in replacement for torchvision transforms.

    ## Example

    ```python
    imgs = [Image.open(f"image{i}.png") for i in range(1, 4)]
    t = get_transforms(train=True)
    t_imgs = t(imgs) # List[torch.Tensor]
    ```

    For the single image case:

    ```python
    img = Image.open(f"image{0}.png")
    # or img = np.load(some_bytes)
    t = get_transforms(train=True)
    t_img = t(img) # torch.Tensor
    ```
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    _data_transform = None

    def _get_transform(n: int = 3):
        if train:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.RandomResizedCrop(224, 224, scale=(0.2, 1.0)),
                    A.HorizontalFlip(),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        else:
            data_transforms = A.Compose(
                [
                    A.Resize(224, 224),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ],
                additional_targets={f"image{i}": "image" for i in range(1, n)},
            )
        return data_transforms

    def transform_images(images: any):
        nonlocal _data_transform

        if not isinstance(images, list):
            n = 1
            images = [images]
        else:
            n = len(images)
        if _data_transform is None:
            # instantiate once
            _data_transform = _get_transform(n)

        # accepts both lists of np.Array and PIL.Image
        if isinstance(images[0], Image.Image):
            images = [np.array(img) for img in images]

        image_dict = {"image": images[0]}
        for i in range(1, n):
            image_dict[f"image{i}"] = images[i]

        transformed = _data_transform(**image_dict)
        transformed_images = [
            transformed[key] for key in transformed.keys() if "image" in key
        ]

        if len(transformed_images) == 1:
            return transformed_images[0]
        return transformed_images

    return transform_images
