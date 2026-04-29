"""File which contains function related to
conversion into specific format such as zarr"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

import numpy as np
import zarr

from cd_mm_sits.constant.data import REF_DATE


@dataclass
class OutputArray:
    sits: np.ndarray
    label: np.ndarray
    mask: np.ndarray
    time: np.ndarray
    orbites: np.ndarray | None = None


def from_dict_2_array(
    dict_data: dict,
    sorted_times: list,
    base_date: str = REF_DATE,
    mod: Literal["s1", "s2"] = "s2",
) -> OutputArray:
    year, month, day = base_date.split("-")
    base_date = date(int(year), int(month), int(day))
    sits = np.stack([dict_data[date].raster for date in sorted_times], axis=0)
    label = np.stack([dict_data[date].label for date in sorted_times], axis=0)
    time = np.array(
        [
            int(
                (datetime.strptime(date_str, "%Y-%m-%d").date() - base_date)
                / np.timedelta64(1, "D")
            )
            for date_str in sorted_times
        ]
    )
    if mod == "s1":
        orbites = np.stack([dict_data[date].orbite for date in sorted_times], axis=0)
    elif mod == "s2":
        orbites = None
    else:
        raise NotImplementedError
    mask = np.stack([dict_data[date].mask for date in sorted_times], axis=0)
    return OutputArray(sits=sits, label=label, mask=mask, time=time, orbites=orbites)


def build_one_mode_group(
    grp: zarr.Group,
    dict_data: dict,
    sorted_times: list,
    chunk_size: int = 128,
    mod: Literal["s1", "s2"] = "s2",
) -> zarr.Group | None:
    output_array = from_dict_2_array(
        dict_data=dict_data, sorted_times=sorted_times, mod=mod
    )
    t, c, h, w = output_array.sits.shape
    if (h < chunk_size) or (w < chunk_size):
        return None
    arr = grp.create_array(
        "sits",
        shape=output_array.sits.shape,
        chunks=(1, c, chunk_size, chunk_size),
        dtype=output_array.sits.dtype,
    )
    arr[:] = output_array.sits

    arr = grp.create_array(
        "label",
        shape=output_array.label.shape,
        chunks=(1, 128, 128),
        dtype=output_array.label.dtype,
    )

    arr[:] = output_array.label
    # print(label_mask.pshape)
    arr = grp.create_array(
        "data_mask",
        shape=output_array.mask.shape,
        chunks=(1, 128, 128),
        dtype=output_array.mask.dtype,
    )
    arr[:] = output_array.mask

    arr = grp.create_array("date", shape=(t), chunks=(1,), dtype=np.int32)
    arr[:] = output_array.time

    if output_array.orbites is not None:
        arr = grp.create_array(
            "relative_orbites", shape=(t), chunks=(1,), dtype=np.int32
        )
        arr[:] = output_array.orbites
    return grp


def create_mm_zarr(
    path_zarr: str,
    s2_dict_data: dict,
    s1_dict_data: dict,
    site_mask: np.ndarray | None,
    chunk_size: int = 128,
):
    """create a zarr file for one mulitmodal SITS"""
    root = zarr.open(path_zarr, mode="w")
    grp_s2 = root.create_group("S2")
    grp_s2 = build_one_mode_group(
        grp=grp_s2,
        dict_data=s2_dict_data,
        sorted_times=sorted(s2_dict_data.keys()),
        chunk_size=chunk_size,
        mod="s2",
    )
    if grp_s2 is None:
        return None
    grp_s1 = root.create_group("S1")
    grp_s1 = build_one_mode_group(
        grp=grp_s1,
        dict_data=s1_dict_data,
        sorted_times=sorted(s1_dict_data.keys()),
        chunk_size=chunk_size,
        mod="s1",
    )
    if site_mask is not None:
        grp_mask_site = root.create_group("mask_site")
        arr = grp_mask_site.create_array(
            "raster", shape=site_mask.shape, chunks=(chunk_size, chunk_size), dtype=bool
        )
        arr[:] = site_mask
    return root
