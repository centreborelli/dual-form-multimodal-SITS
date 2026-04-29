"""File for loading the solar panel dataset"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from cd_mm_sits.constant.data import REF_DATE
from cd_mm_sits.data.dataset.sp_s2_zarr import SPSampleSITS
from cd_mm_sits.data.dataset.utils import (
    perform_center_crop,
    perform_random_crop,
    temporal_crop,
)


class SolarPanel(Dataset):
    def __init__(
        self,
        path_dataset: str,
        list_id: list,
        transform: nn.Module | None = None,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
    ) -> None:
        super().__init__()
        self.path_dataset = path_dataset
        self.list_id = list_id
        self.reference_date = REF_DATE
        self.crop_size = crop_size
        self.transform = transform
        self.max_len = max_len
        self.dataset_type = dataset_type

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
        sits = Tensor(
            rearrange(
                np.load(
                    Path(self.path_dataset).joinpath(f"{extracted_id}_l1c.npy"),
                ),
                "t h w c -> c t  h w",
            )
        )

        # Normalized the SITS
        if self.transform is not None:
            sits = self.transform(sits)
        sits = rearrange(sits, " c t h w -> t c h w")
        time = np.load(
            Path(self.path_dataset).joinpath(f"{extracted_id}_timestamp.npy"),
        )
        label = np.load(Path(self.path_dataset).joinpath(f"{extracted_id}_mask.npy"))
        out_temporal_crop = temporal_crop(
            sits, time, label, max_len=self.max_len, seed=seed
        )
        sits = out_temporal_crop.x
        time = out_temporal_crop.time
        label = out_temporal_crop.label
        if self.dataset_type == "train":
            sits, label = perform_random_crop(
                Tensor(sits), Tensor(label), (self.crop_size, self.crop_size)
            )
        else:
            sits, label = perform_center_crop(
                Tensor(sits), Tensor(label), (self.crop_size, self.crop_size)
            )

        return SPSampleSITS(
            sits=sits,
            label=label,
            time=Tensor(time),
            extracted_id=extracted_id,
        )
