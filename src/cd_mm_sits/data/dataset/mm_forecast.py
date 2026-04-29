from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
import ziptif.loader
import ziptif.ziptif
from torch.utils.data import Dataset

from cd_mm_sits.constant.dataset import AGERA5_VAR, S2_BANDS_10M, S2_BANDS_20M
from cd_mm_sits.data.datamodule.time_sampling import (
    RANDOM_SAMPLING_STRATEGY,
    TimeSamplingStrategy,
)
from cd_mm_sits.data.dataset.ziptif_dataset import (
    S1ForcSITSSample,
    S2ForcSITSSample,
    extract_agera_5_feat,
    get_random_roi,
    s1_timeseries_to_sits,
    s2_timeseries_to_sits,
)


@dataclass(frozen=True)
class MMForcSample:
    s1: S1ForcSITSSample
    s2: S2ForcSITSSample


class MMForecastDataset(Dataset):
    def __init__(
        self,
        site_collection_s1: ziptif.ziptif.SiteCollection,
        site_collection_s2: ziptif.ziptif.SiteCollection,
        list_id: list,
        fsmap: ziptif.loader.FileSystemMapping,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
        time_sampling: TimeSamplingStrategy = RANDOM_SAMPLING_STRATEGY,
        path_agera_5: str | None = None,
        weather_past: int = 10,  # past time step in weather
        weather_vars: list | None = None,
    ) -> None:
        super().__init__()

        if crop_size > 128 or crop_size % 2 != 0:
            # if we wanted we could support >128, but this is not supported right now
            raise ValueError("crop_size must be <= 128 and even")

        self.site_collection_s1 = site_collection_s1
        self.site_collection_s2 = site_collection_s2
        self.list_id = list_id
        self.crop_size = crop_size
        self.max_len = max_len
        self.dataset_type = dataset_type
        self.fsmap = fsmap
        self.openedzips = None
        self.rand_train = None
        self.time_sampling = time_sampling
        self.weather_past = weather_past
        self.path_agera_5 = path_agera_5
        if self.path_agera_5 is not None:
            if weather_vars is None:
                self.weather_vars = AGERA5_VAR
            else:
                self.weather_vars = self.weather_vars
            assert Path(self.path_agera_5).exists(), f"{self.path_agera_5} not found"
            agera_5_df = pl.read_parquet(self.path_agera_5, parallel="none")
            self.agera_5_df = agera_5_df.with_columns(
                (pl.col(*self.weather_vars) - pl.col(*self.weather_vars).mean())
                / pl.col(*self.weather_vars).std()
            )
        print(f"Loaded weather variables: \n {self.path_agera_5} ")

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index: int) -> MMForcSample:
        if not self.openedzips:
            self.openedzips = self.fsmap.open_all()
        if not self.rand_train:
            self.rand_train = random.Random()

        if self.dataset_type == "train":
            rand = self.rand_train
        else:
            rand = random.Random(index)

        site_id = self.list_id[index]
        site_s1 = self.site_collection_s1.get_site(site_id)
        site_s2 = self.site_collection_s2.get_site(site_id)

        n_max = min(self.max_len, site_s1.n_dates, site_s2.n_dates)
        indices = self.time_sampling.sample(
            {
                "s1": site_s1.acquisitions_metadata,
                "s2": site_s2.acquisitions_metadata,
            },
            n_max,
            rand,
        )
        indices_s1 = sorted(indices["s1"])
        indices_s2 = sorted(indices["s2"])

        roi = get_random_roi(self.crop_size, rand)

        req_s2_10m = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s2,
            site_id=site_id,
            time_indices=indices_s2,
            bands=S2_BANDS_10M,
            roi=roi,
        )
        req_s2_20m = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s2,
            site_id=req_s2_10m.site_id,
            time_indices=req_s2_10m.time_indices,
            bands=S2_BANDS_20M,
            roi=(roi[0] // 2, roi[1] // 2, roi[2] // 2, roi[3] // 2),
        )

        req_s1 = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s1,
            site_id=site_id,
            time_indices=indices_s1,
            bands=["VV", "VH"],
            roi=roi,
        )

        req = ziptif.loader.BundleReadRequest(
            subrequests={
                "S2_10m": req_s2_10m,
                "S2_20m": req_s2_20m,
                "S1": req_s1,
            },
        )

        result = ziptif.loader.execute_bundle_read_request(
            req, openedzips=self.openedzips
        )

        s1_sits = s1_timeseries_to_sits(result.timeseries["S1"])
        s2_sits = s2_timeseries_to_sits(
            result.timeseries["S2_10m"], result.timeseries["S2_20m"]
        )

        if self.agera_5_df is not None:
            days_s2 = result.timeseries["S2_10m"].stac_metadata["datetime"]
            days_s1 = result.timeseries["S1"].stac_metadata["datetime"]
            aux_s2 = extract_agera_5_feat(
                self.agera_5_df,
                times=days_s2,
                patch_id=site_id,
                weather_col=self.weather_vars,
                past_days=self.weather_past,
            )
            aux_s1 = extract_agera_5_feat(
                self.agera_5_df,
                times=days_s1,
                patch_id=site_id,
                weather_col=self.weather_vars,
                past_days=self.weather_past,
            )
            s2_sits = S2ForcSITSSample(
                content=s2_sits.content,
                time=s2_sits.time,
                validity_mask=s2_sits.validity_mask,
                weather_var=aux_s2.to(dtype=s2_sits.content.dtype),
            )
            s1_sits = S1ForcSITSSample(
                content=s1_sits.content,
                time=s1_sits.time,
                validity_mask=s1_sits.validity_mask,
                weather_var=aux_s1,
                angles=s1_sits.angles,
                orbits=s1_sits.orbits,
            )
        sits = MMForcSample(
            s1=s1_sits,
            s2=s2_sits,
        )

        return sits


class MMTestForecastDataset(MMForecastDataset):
    def __init__(
        self,
        site_collection_s1: ziptif.ziptif.SiteCollection,
        site_collection_s2: ziptif.ziptif.SiteCollection,
        list_id: list,
        fsmap: ziptif.loader.FileSystemMapping,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
        time_sampling: TimeSamplingStrategy = RANDOM_SAMPLING_STRATEGY,
        path_agera_5: str | None = None,
        weather_past: int = 10,  # past time step in weather
        weather_vars: list | None = None,
    ) -> None:
        super().__init__(
            site_collection_s1=site_collection_s1,
            site_collection_s2=site_collection_s2,
            list_id=list_id,
            fsmap=fsmap,
            dataset_type=dataset_type,
            max_len=max_len,
            crop_size=crop_size,
            time_sampling=time_sampling,
            path_agera_5=path_agera_5,
            weather_past=weather_past,
            weather_vars=weather_vars,
        )

    def __getitem__(self, index: int) -> MMForcSample:
        if not self.openedzips:
            self.openedzips = self.fsmap.open_all()
        if not self.rand_train:
            self.rand_train = random.Random()

        if self.dataset_type == "train":
            rand = self.rand_train
        else:
            rand = random.Random(index)

        site_id = self.list_id[index]
        site_s1 = self.site_collection_s1.get_site(site_id)
        site_s2 = self.site_collection_s2.get_site(site_id)

        n_max = min(self.max_len, site_s1.n_dates, site_s2.n_dates)
        indices_s1 = [i for i in range(site_s1.n_dates)][
            -n_max:
        ]  # takes the last max_len values
        indices_s2 = [i for i in range(site_s2.n_dates)][-n_max:]

        roi = get_random_roi(self.crop_size, rand)

        req_s2_10m = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s2,
            site_id=site_id,
            time_indices=indices_s2,
            bands=S2_BANDS_10M,
            roi=roi,
        )
        req_s2_20m = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s2,
            site_id=req_s2_10m.site_id,
            time_indices=req_s2_10m.time_indices,
            bands=S2_BANDS_20M,
            roi=(roi[0] // 2, roi[1] // 2, roi[2] // 2, roi[3] // 2),
        )

        req_s1 = ziptif.loader.ReadRequest(
            site_collection=self.site_collection_s1,
            site_id=site_id,
            time_indices=indices_s1,
            bands=["VV", "VH"],
            roi=roi,
        )

        req = ziptif.loader.BundleReadRequest(
            subrequests={
                "S2_10m": req_s2_10m,
                "S2_20m": req_s2_20m,
                "S1": req_s1,
            },
        )

        result = ziptif.loader.execute_bundle_read_request(
            req, openedzips=self.openedzips
        )

        s1_sits = s1_timeseries_to_sits(result.timeseries["S1"])
        s2_sits = s2_timeseries_to_sits(
            result.timeseries["S2_10m"], result.timeseries["S2_20m"]
        )

        if self.agera_5_df is not None:
            days_s2 = result.timeseries["S2_10m"].stac_metadata["datetime"]
            days_s1 = result.timeseries["S1"].stac_metadata["datetime"]
            aux_s2 = extract_agera_5_feat(
                self.agera_5_df,
                times=days_s2,
                patch_id=site_id,
                weather_col=self.weather_vars,
                past_days=self.weather_past,
            )
            aux_s1 = extract_agera_5_feat(
                self.agera_5_df,
                times=days_s1,
                patch_id=site_id,
                weather_col=self.weather_vars,
                past_days=self.weather_past,
            )
            s2_sits = S2ForcSITSSample(
                content=s2_sits.content,
                time=s2_sits.time,
                validity_mask=s2_sits.validity_mask,
                weather_var=aux_s2.to(dtype=s2_sits.content.dtype),
            )
            s1_sits = S1ForcSITSSample(
                content=s1_sits.content,
                time=s1_sits.time,
                validity_mask=s1_sits.validity_mask,
                weather_var=aux_s1,
                angles=s1_sits.angles,
                orbits=s1_sits.orbits,
            )
        sits = MMForcSample(
            s1=s1_sits,
            s2=s2_sits,
        )

        return sits
