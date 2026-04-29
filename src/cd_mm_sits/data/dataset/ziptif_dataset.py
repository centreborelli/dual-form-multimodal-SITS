import abc
import datetime
import random
from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
import ziptif.loader
from einops import rearrange
from torch import Tensor

from cd_mm_sits.constant.data import REF_DATE


@dataclass(frozen=True)
class BaseForcSITSSample(abc.ABC):
    content: Tensor
    time: Tensor
    validity_mask: Tensor
    weather_var: Tensor | None = None  # is there a better way
    angles: Tensor | None = None


@dataclass(frozen=True)
class S1ForcSITSSample(BaseForcSITSSample):
    orbits: Tensor | None = None


@dataclass(frozen=True)
class S2ForcSITSSample(BaseForcSITSSample):
    pass


def extract_time_from_metadata(metadata: pl.DataFrame) -> Tensor:
    ref_date = datetime.datetime.strptime(REF_DATE, "%Y-%m-%d").replace(
        tzinfo=datetime.UTC
    )
    time = [
        (t - ref_date).days + (t - ref_date).seconds / 60 / 60 / 24
        for t in metadata["datetime"]
    ]
    time = torch.tensor(time, dtype=torch.float32)
    assert torch.all(torch.diff(time) >= 0)
    return time


def extract_orbits_from_s1_metadata(metadata: pl.DataFrame) -> Tensor:
    orbits = metadata["sentinel1:relative_orbit_number"]
    orbits = torch.tensor(orbits, dtype=torch.int)
    return orbits


def extract_angles_from_s1_metadata(
    metadata: pl.DataFrame,
) -> Tensor:  # azimuth + elevation
    angles_azimuth = torch.deg2rad(torch.tensor(metadata["ard:azimuth_angle"]))
    angles_incidence = torch.deg2rad(torch.tensor(metadata["ard:incidence_angle"]))
    return torch.stack([angles_azimuth, angles_incidence], dim=-1)


def s1_timeseries_to_sits(timeseries: ziptif.loader.Timeseries) -> S1ForcSITSSample:
    all_bands = [torch.from_numpy(timeseries.bands[b]) for b in timeseries.bands]
    content = torch.concatenate(all_bands, dim=1)

    mask = ~torch.isnan(content).any(dim=1)

    # due to S1 calibration and resampling, we might have <0 values
    # but since the signal represents amplitudes, we clamp to a small positive value
    # because we might later have to assume non-negativity (e.g., when converting to dB)
    torch.clamp_(content, min=1e-7)
    torch.nan_to_num_(content, nan=0.0)

    metadata: pl.DataFrame = timeseries.stac_metadata
    time = extract_time_from_metadata(metadata)
    orbits = extract_orbits_from_s1_metadata(metadata)
    angles = extract_angles_from_s1_metadata(metadata)
    if torch.is_anomaly_check_nan_enabled():
        assert torch.all(content >= 0.0)
        assert not torch.any(torch.isnan(time))
        assert not torch.any(torch.isnan(orbits))
        assert not torch.any(torch.isnan(mask))
        assert not torch.any(torch.isnan(content))
        assert not torch.any(torch.isnan(angles))
    return S1ForcSITSSample(
        content=content, time=time, validity_mask=mask, orbits=orbits, angles=angles
    )


def s2_timeseries_to_sits(
    timeseries_10m: ziptif.loader.Timeseries,
    timeseries_20m: ziptif.loader.Timeseries,
) -> S2ForcSITSSample:
    def to_torch(a):  # TODO: convert to float from on_after_batch_transfer
        return torch.from_numpy(a).float()

    all_bands = [
        to_torch(timeseries_10m.bands[b]) for b in timeseries_10m.bands if b != "KLD"
    ]
    for b in timeseries_20m.bands:
        input = to_torch(timeseries_20m.bands[b])
        arr = torch.nn.functional.interpolate(
            input,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        if False:
            arr_ref = torch.nn.functional.interpolate(
                to_torch(timeseries_20m.bands[b]),
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            assert (arr == arr_ref).all()
        all_bands.append(arr)

    content = torch.concatenate(all_bands, dim=1)

    metadata: pl.DataFrame = timeseries_10m.stac_metadata

    time = extract_time_from_metadata(metadata)

    kld = timeseries_10m.bands["KLD"][:, 0, :, :]
    mask = torch.from_numpy(kld == 0)

    if torch.is_anomaly_check_nan_enabled():
        assert not torch.any(torch.isnan(time))
        assert not torch.any(torch.isnan(mask))
        assert not torch.any(torch.isnan(content))
    return S2ForcSITSSample(
        content=content,
        time=time,
        validity_mask=mask,
    )


def get_random_roi(crop_size: int, rand: random.Random) -> tuple[int, int, int, int]:
    """for now, this function assumes a 128x128 tiles inside a 512x512 image"""
    base_x = rand.randint(0, 3) * 128
    base_y = rand.randint(0, 3) * 128
    roi = (
        rand.randint(base_x, base_x + 128 - crop_size),
        rand.randint(base_y, base_y + 128 - crop_size),
        crop_size,
        crop_size,
    )
    # snap to even coordinates for the 20m bands
    roi = (roi[0] // 2 * 2, roi[1] // 2 * 2, roi[2], roi[3])
    assert roi[0] % 2 == 0, (roi[0], "must be even")
    assert roi[1] % 2 == 0, (roi[1], "must be even")
    assert roi[2] % 2 == 0, (roi[2], "must be even")
    assert roi[3] % 2 == 0, (roi[3], "must be even")
    return roi


def compute_past_weather(
    final_df: pl.DataFrame, all_vars: list, past_days: int
) -> tuple[pl.DataFrame, list]:
    weather_var = []
    for var in all_vars:
        var_name = f"{var}_{past_days}days"
        final_df = final_df.sort("time").with_columns(
            [
                pl.concat_list(
                    [pl.col(var).shift(i) for i in range(0, past_days)]
                ).alias(var_name)
            ]
        )
        weather_var += [var_name]
    return final_df, weather_var


def extract_agera_5_feat(
    df: pl.DataFrame, patch_id: str, times: Tensor, weather_col: list, past_days: int
) -> Tensor:
    all_vars = []
    sub_df = df.filter(pl.col("id") == patch_id).sort("time").unique(subset=["time"])
    for var in weather_col:
        sub_df = sub_df.with_columns(
            [
                pl.concat_list(
                    [pl.col(var).shift(i) for i in range(0, past_days)]
                ).alias(f"{var}_{past_days}days")
            ]
        )
        all_vars += [f"{var}_{past_days}days"]

    times_df = pl.DataFrame(
        {"target_date": [dt.date() for dt in times], "order": range(len(times))}
    )

    # Join with your processed sub_df
    sub_df_with_date = sub_df.with_columns(
        pl.col("time").dt.date().alias("target_date")
    )
    result_df = times_df.join(sub_df_with_date, on="target_date", how="left").sort(
        "order"
    )
    assert len(result_df) == times.shape[0], len(result_df)
    arr = np.stack(
        [np.stack(result_df[var].to_list()).astype(np.float32) for var in all_vars]
    )
    assert arr.shape[1] == times.shape[0], f"ar {arr.shape} times {times.shape}"
    return rearrange(torch.from_numpy(arr), "D T Tbef -> T Tbef D")  # D=nvar
