import random
from dataclasses import dataclass

import torch
import torchvision
from torch import Tensor
from torch.types import Number

from cd_mm_sits.constant.label import IGNORE_LABEL


@dataclass
class OutTemporalCrop:
    x: Tensor
    time: Tensor
    label: Tensor
    mask: Tensor | None = None
    orbites: Tensor | None = None


def temporal_crop(
    x: Tensor,
    time: Tensor,
    label: Tensor,
    max_len: int,
    seed: int | None = None,
    mask: Tensor | None = None,
    orbites: Tensor | None = None,
) -> OutTemporalCrop:
    """Perform the temporal crop of a SITS
    x: Tensor T,C,H,W
    max_len:
    :returns:
    - a tensor of shape min(max_len,T),C, H, W
    - a tensor of shape min(max_len,T)
    - a tensor of shape min(max_len,T),H,W
    """
    T = x.shape[0]
    gen = torch.Generator()
    if seed is None:
        seed = random.randint(1, 10000)
    gen.manual_seed(seed)
    # T,H,W
    # print(label)

    label = torch.as_tensor(label)
    relevant_dates = torch.nonzero(
        torch.sum(torch.sum(label == IGNORE_LABEL, dim=-1), dim=-1) == 0
    )

    if max_len < T:
        idx = torch.randperm(T, generator=gen)[:max_len]
        if relevant_dates.shape[0] > 0:
            include_label_idx = relevant_dates[
                torch.randperm(relevant_dates.shape[0], generator=gen)[0]
            ]
            if include_label_idx not in idx:
                idx[0] = include_label_idx
        idx = idx.sort()[0].numpy()
    else:
        idx = [i for i in range(T)]

    if mask is not None:
        mask = mask[idx, ...]
    if orbites is not None:
        orbites = orbites[idx, ...]

    return OutTemporalCrop(
        x=x[idx, ...],
        time=time[idx, ...],
        label=label[idx, ...],
        mask=mask,
        orbites=orbites,
    )


def get_params_crop(
    img: Tensor, output_size: tuple[int, int]
) -> tuple[int | Number, int | Number, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    *_, h, w = torchvision.transforms.functional.get_dimensions(img)
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(
            f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
        )

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw


def perform_random_crop(
    sits: Tensor, label: Tensor, output_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """
    :param sits: T, C, H, W
    :param label: T,H,W
    returns
    - T,C,output_size[0],output_size[1]
    - T,output_size[0],output_size[1]
    """

    i, j, h, w = get_params_crop(sits, output_size=output_size)
    return torchvision.transforms.functional.crop(
        sits, i, j, h, w
    ), torchvision.transforms.functional.crop(label, i, j, h, w)


def perform_center_crop(sits: Tensor, label: Tensor, output_size: tuple[int, int]):
    return torchvision.transforms.functional.center_crop(
        sits, output_size
    ), torchvision.transforms.functional.center_crop(label, output_size)


def get_rd_index_crop(
    h: int, w: int, output_size: tuple[int, int]
) -> tuple[int | Number, int | Number, int, int]:
    """Get parameters for ``crop`` for a random crop.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (tuple): Expected output size of the crop.

    Returns:
        tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
    """
    th, tw = output_size

    if h < th or w < tw:
        raise ValueError(
            f"Required crop size {(th, tw)} is larger than input image size {(h, w)}"
        )

    if w == tw and h == th:
        return 0, 0, h, w

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return i, j, th, tw


def get_center_crop_bounds(image_size, patch_size) -> tuple[int, int, int, int]:
    """
    Compute the bounding coordinates for a center crop.

    Args:
        image_size (tuple): (height, width) of the full image.
        patch_size (tuple): (crop_height, crop_width) for the patch.

    Returns:
        tuple: (top, left, patch_height,patch width ) coordinates of the crop.
    """
    image_height, image_width = image_size
    patch_height, patch_width = patch_size

    if patch_height > image_height or patch_width > image_width:
        raise ValueError("Patch size must be smaller than or equal to the image size.")

    top = (image_height - patch_height) // 2
    left = (image_width - patch_width) // 2

    return top, left, patch_height, patch_width
