"""File for the dataset for the multimodal solar panel
datasets, saved as .zarr format"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import zarr
from torch import Tensor
from torch.utils.data import Dataset

from cd_mm_sits.constant.data import REF_DATE
from cd_mm_sits.data.dataset.sp_s2_zarr import SPSampleSITS, read_zarr_mod
from cd_mm_sits.data.dataset.utils import get_center_crop_bounds, get_rd_index_crop


@dataclass
class SPMMSampleSits:
    s1: SPSampleSITS
    s2: SPSampleSITS
    mask_site: Tensor | None = None


@dataclass
class MMMaxLen:
    s2: None | int = None
    s1: None | int = None


@dataclass
class MMTransform:
    s2: nn.Module | None = None
    s1: nn.Module | None = None


class ZarrMMSolarPanel(Dataset):
    def __init__(
        self,
        path_dataset: str,
        list_id: list,
        transform: MMTransform | None = None,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: MMMaxLen | None = None,
        crop_size: int = 64,
        is_mask_site: bool = True,
    ) -> None:
        super().__init__()
        self.path_dataset = path_dataset
        self.list_id = list_id  # the list of the site id
        self.reference_date = REF_DATE
        self.crop_size = crop_size
        if transform is None:
            transform = MMTransform()
        self.transform = transform
        if max_len is None:
            max_len = MMMaxLen()
        self.max_len = max_len
        self.dataset_type = dataset_type
        self.is_mask_site = is_mask_site

    def __post_init__(self):
        if self.dataset_type in ["val", "test"]:
            assert self.max_len.s2 is None
            assert self.max_len.s1 is None

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index: int) -> SPMMSampleSits:
        """Read the MM SITS for solar panel task

        :param index:
        :returns:

        """
        if self.dataset_type == "train":
            seed = None
        else:
            seed = index
        extracted_id = self.list_id[index]

        zarr_mm = zarr.open(
            Path(self.path_dataset).joinpath(f"{extracted_id}.zarr"), mode="r"
        )
        if self.is_mask_site:
            mask_site = zarr_mm["mask_site"]["raster"]
            # print(mask_site.shape)
        else:
            mask_site = None
        _, H, W = zarr_mm["S2"]["label"].shape
        if self.dataset_type == "train":
            i, j, th, tw = get_rd_index_crop(
                h=H, w=W, output_size=(self.crop_size, self.crop_size)
            )
        else:
            i, j, th, tw = get_center_crop_bounds(
                image_size=(H, W), patch_size=(self.crop_size, self.crop_size)
            )
        if mask_site is not None:
            mask_site = torch.Tensor(mask_site[i : i + th, j : j + tw])

        s1_data = read_zarr_mod(
            zarr_mod=zarr_mm["S1"],
            h=int(i),
            w=int(j),
            th=th,
            tw=tw,
            seed=seed,
            max_len=self.max_len.s1,
            transform=self.transform.s1,
            mod="s1",
        )
        s2_data = read_zarr_mod(
            zarr_mod=zarr_mm["S2"],
            h=int(i),
            w=int(j),
            th=th,
            tw=tw,
            seed=seed,
            max_len=self.max_len.s2,
            transform=self.transform.s2,
            mod="s2",
        )
        return SPMMSampleSits(
            s1=s1_data,
            s2=s2_data,
            mask_site=mask_site,
        )
