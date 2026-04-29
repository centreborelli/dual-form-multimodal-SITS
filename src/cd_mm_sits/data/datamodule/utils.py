"""
FIle for functions relevant to transform applied on batch
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import ziptif.loader
import ziptif.ziptif
from einops import rearrange
from torch import Tensor, nn

from cd_mm_sits.constant.dataset import SP_FOLD_PLANNING
from cd_mm_sits.data.datamodule.transform import (
    Clip,
    OneTransform,
    S2BatchNormalize,
    S2Normalize,
    Stats,
)


def read_csv_stat(path_csv) -> Stats:
    """
    Read the stats stored in a casv
    Parameters
    ----------
    path_csv :

    Returns
    -------

    """
    assert path_csv.exists(), f"No file found at {path_csv}"
    df_stats = pd.read_csv(path_csv, sep=",", index_col=0)
    return Stats(
        median=df_stats["med"].tolist(),
        qmin=df_stats["qmin"].tolist(),
        qmax=df_stats["qmax"].tolist(),
    )


def load_transform_one_mod(
    path_dir_csv: str, mod: Literal["s1", "s2", "s2_batch"] = "s2"
) -> OneTransform:
    """

    Parameters
    ----------
    path_dir_csv : path where "dataset_{mod}.csv" is stored,
     which contains modality stats
    mod : modality, implemented only of S2

    Returns
    -------

    """
    if mod == "s2_batch":
        path_csv = Path(path_dir_csv).joinpath("dataset_s2.csv")
    else:
        path_csv = Path(path_dir_csv).joinpath(f"dataset_{mod}.csv")
    stats = read_csv_stat(path_csv)

    scale = tuple(
        [float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin, strict=False)]
    )
    if mod == "s2":
        transform_mod = OneTransform(
            torch.nn.Sequential(
                Clip(qmin=stats.qmin, qmax=stats.qmax),
                S2Normalize(med=stats.median, scale=scale),
            ),
            stats,
        )
        return transform_mod
    if mod == "s2_batch":
        transform_mod = OneTransform(
            torch.nn.Sequential(
                S2BatchNormalize(med=stats.median, scale=scale),
            ),
            stats,
        )
        return transform_mod
    elif mod == "s1":
        transform_mod = OneTransform(
            torch.nn.Sequential(
                S2Normalize(med=stats.median, scale=scale),
            ),
            stats,
        )
        return transform_mod
    else:
        raise NotImplementedError


def apply_transform_basic(batch_sits: Tensor, transform: nn.Module, bs: int) -> Tensor:
    """
    Reshape before applying transform
    Parameters
    ----------
    batch_sits :
    transform :

    Returns
    -------

    """
    batch_sits = rearrange(batch_sits, " b t c h w -> c (b t) h w")
    batch_sits = transform(batch_sits)
    batch_sits = rearrange(batch_sits, "c (b t )  h w -> b t c h w", b=bs)
    return batch_sits


def split_dataset(
    data: list,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed=None,
):
    """
    Split a list into three sublists
    """
    if not (0 < train_ratio + val_ratio + test_ratio <= 1.0):
        raise ValueError("Ratios must sum to 1.0 or less.")

    if seed is not None:
        random.seed(seed)

    data_copy = data[:]
    random.shuffle(data_copy)

    total = len(data_copy)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    train_data = data_copy[:train_end]
    val_data = data_copy[train_end:val_end]
    test_data = data_copy[val_end:]

    return train_data, val_data, test_data


def read_sites(paths: list, dataset_path: str) -> list:
    list_id = []
    for path in paths:
        with open(Path(dataset_path).joinpath(path)) as f:
            txt = f.read()
            list_id += txt.split("\n")[:-1]
    return list_id


def read_txt_fold(fold_expe: int, dataset_path) -> tuple[list, list, list]:
    return (
        read_sites(SP_FOLD_PLANNING[fold_expe]["train"], dataset_path=dataset_path),
        read_sites(SP_FOLD_PLANNING[fold_expe]["val"], dataset_path=dataset_path),
        read_sites(SP_FOLD_PLANNING[fold_expe]["test"], dataset_path=dataset_path),
    )


@dataclass(frozen=True)
class ZipTifPaths:
    index_paths: dict[str, list[str]]
    """ modality -> paths to indexes """
    zip_paths: dict[ziptif.loader.ZipId, str]
    """ zip_id -> path to zip file """

    def read_site_collection(self, modality: str) -> ziptif.ziptif.SiteCollection:
        bytes = []
        for index_path in self.index_paths[modality]:
            with open(index_path, "rb") as f:
                bytes.append(f.read())
        site_collection = ziptif.ziptif.SiteCollection.from_bytes_multiple(bytes)
        return site_collection

    def get_fsmap(self) -> ziptif.loader.FileSystemMapping:
        return ziptif.loader.FileSystemMapping(self.zip_paths)
