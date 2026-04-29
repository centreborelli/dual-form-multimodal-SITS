"""File with the class dataset associated to loading S2 on SITS on zarr format"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import zarr
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from cd_mm_sits.constant.data import REF_DATE
from cd_mm_sits.data.dataset.utils import (
    get_center_crop_bounds,
    get_rd_index_crop,
    temporal_crop,
)


def read_zarr_mod(
    zarr_mod,
    h: int,
    w: int,
    th: int,
    tw: int,
    max_len: int | None = None,
    transform: nn.Module | None = None,
    mod: Literal["s2", "s1"] = "s2",
    seed: int | None = None,
) -> SPSampleSITS:
    sits = zarr_mod["sits"][..., h : h + th, w : w + tw]
    label = zarr_mod["label"][..., h : h + th, w : w + tw]
    label[label >= 1] = 1
    time = zarr_mod["date"][:]
    mask = zarr_mod["data_mask"][..., h : h + th, w : w + tw]
    if mod == "s1":
        orbites = zarr_mod["relative_orbites"][:]
    else:
        orbites = None
    if max_len is not None:
        item_input = temporal_crop(
            x=sits,
            time=time,
            label=label,
            mask=mask,
            max_len=max_len,
            seed=seed,
            orbites=orbites,
        )
        sits = item_input.x
        label = item_input.label
        time = item_input.time
        orbites = item_input.orbites
        if item_input.mask is not None:
            mask = torch.from_numpy(item_input.mask[:])
    else:
        label = torch.from_numpy(label)
    if orbites is not None:
        orbites = torch.from_numpy(orbites[:])
    if transform is not None:
        sits = torch.from_numpy(sits[:]).float()
        sits = transform(rearrange(sits, "t c h w -> c t h w"))
        sits = rearrange(sits, "c t h w -> t c h w")
        sits = torch.nan_to_num(sits, 0)
    else:
        sits = torch.from_numpy(sits[:])
        sits = torch.nan_to_num(sits, 0)
    return SPSampleSITS(
        sits=sits,
        label=label,
        time=torch.from_numpy(time[:]),
        mask=mask,
        orbites=orbites,
    )


@dataclass
class SPSampleSITS:
    sits: Tensor  # T,C,H,W
    label: Tensor  # T H W
    time: Tensor  # T
    mask: Tensor | None = None
    extracted_id: str | None = None
    orbites: Tensor | None = None
    mask_site: Tensor | None = None

    def __post_init__(self):
        # check time dimension matches
        assert self.sits.shape[0] == self.label.shape[0]
        assert self.label.shape[0] == self.time.shape[0]
        if self.mask is not None:
            assert self.sits.shape[0] == self.mask.shape[0]
        if self.orbites is not None:
            assert self.orbites.shape[0] == self.sits.shape[0]
            assert torch.max(self.orbites) <= 175, f" orbites are {self.orbites}"
            assert torch.min(self.orbites) >= 1, f" orbites are {self.orbites}"
        if self.mask_site is not None:
            assert self.mask_site.shape[-1] == self.sits.shape[-1]

    def __repr__(self):
        dt_sits = str(self.sits.dtype).removeprefix("torch.")
        dt_label = str(self.label.dtype).removeprefix("torch.")
        dt_time = str(self.time.dtype).removeprefix("torch.")
        dt_mask = (
            str(self.mask.dtype).removeprefix("torch.") if self.mask is not None else ""
        )
        dt_orbites = (
            str(self.orbites.dtype).removeprefix("torch.")
            if self.orbites is not None
            else ""
        )
        dt_mask_site = (
            str(self.mask_site.dtype).removeprefix("torch.")
            if self.mask_site is not None
            else ""
        )
        return (
            f"SPSampleSITS(sits={tuple(self.sits.shape)} {dt_sits}, "
            f"label={tuple(self.label.shape)} {dt_label}, "
            f"time={tuple(self.time.shape)} {dt_time}, "
            f"mask={self.mask.shape if self.mask is not None else None} {dt_mask}, "
            f"orbites={tuple(self.orbites.shape) if self.orbites is not None else None} {dt_orbites}, "
            f"mask_site={tuple(self.mask_site.shape) if self.mask_site is not None else None} {dt_mask_site}, "
            f"extracted_id={self.extracted_id})"
        )


class ZarrS2SolarPanel(Dataset):
    def __init__(
        self,
        path_dataset: str,
        list_id: list,
        transform: nn.Module | None = None,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
        is_mask_site: bool = True,
    ) -> None:
        super().__init__()
        self.path_dataset = path_dataset
        self.list_id = list_id
        self.reference_date = REF_DATE
        self.crop_size = crop_size
        self.transform = transform
        self.max_len = max_len
        self.dataset_type = dataset_type
        self.is_mask_site = is_mask_site

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index: int) -> SPSampleSITS:
        """TODO describe function

        :param index:
        :returns:

        """
        if self.dataset_type == "train":
            seed = None
        else:
            seed = index
        extracted_id = self.list_id[index]

        zarr_mm = zarr.open(
            Path(self.path_dataset).joinpath(f"{extracted_id}.zarr"),
        )
        if self.is_mask_site:
            mask_site = zarr_mm["mask_site"]["raster"]
        else:
            mask_site = None
        _, H, W = zarr_mm["S2"]["label"].shape
        # print(mask_site.shape)
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
        s2_data = read_zarr_mod(
            zarr_mod=zarr_mm["S2"],
            h=int(i),
            w=int(j),
            th=th,
            tw=tw,
            seed=seed,
            max_len=self.max_len,
            transform=self.transform,
        )

        return SPSampleSITS(
            sits=s2_data.sits,
            label=s2_data.label,
            time=s2_data.time,
            mask_site=mask_site,
        )
