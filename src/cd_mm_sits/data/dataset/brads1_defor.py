"""File which indicate the pytorch dataset class
related to the Deforestation task using the Bradd-S1-SITS
dataset"""

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import Dataset

from cd_mm_sits.constant.data import REF_DATE
from cd_mm_sits.data.utils import compute_delta_time


@dataclass
class SampleBradS1:
    sits: torch.Tensor  # T C H W
    label: torch.Tensor  # H W
    time: torch.Tensor  # T
    idx_bef: int
    idx_after: int


def find_closest(dates, ref_date):
    """Find the index and day difference of the closest date >= ref_date."""
    idx = bisect_left(dates, ref_date)
    if idx == len(dates):
        idx = -1
    dis = (dates[idx] - ref_date).days
    return idx, dis


class BradS1Dataset(Dataset):
    def __init__(
        self,
        path_dataset: str,
        l_files: list | None = None,
        transform: nn.Module | None = None,
    ):
        super().__init__()
        assert Path(path_dataset).exists()
        self.path_dataset = path_dataset
        self.reference_date = REF_DATE
        if l_files is None:
            self.l_files = [path for path in Path(path_dataset).rglob("*.pt")]
        else:
            self.l_files = l_files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.l_files)

    def __getitem__(self, index: int) -> SampleBradS1:
        input = torch.load(self.l_files[index], weights_only=False)
        sits = input["image"]
        if self.transform is not None:
            sits = rearrange(sits, "T C H W->C T H W")
            sits = rearrange(self.transform(sits), "C T H W -> T C H W")
        time = input["image_dates"]
        corrected_time = [
            compute_delta_time(t, base_date=self.reference_date) for t in time
        ]
        label_time = input["label_dates"]
        idx_before, _ = find_closest(dates=time, ref_date=label_time[0])
        idx_after, _ = find_closest(dates=time, ref_date=label_time[1])
        change_label = input["label"][0, ...] == input["label"][1, ...]
        return SampleBradS1(
            sits=sits,
            time=torch.Tensor(corrected_time),
            label=~change_label,
            idx_after=idx_after,
            idx_bef=idx_before,
        )
