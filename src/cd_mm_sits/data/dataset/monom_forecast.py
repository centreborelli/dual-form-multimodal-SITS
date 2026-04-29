import random
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


class MonoMForecastDataset(Dataset):
    def __init__(
        self,
        site_collection: ziptif.ziptif.SiteCollection,
        list_id: list,
        fsmap: ziptif.loader.FileSystemMapping,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
        mod: Literal["s1", "s2"] = "s2",
        time_sampling: TimeSamplingStrategy = RANDOM_SAMPLING_STRATEGY,
        path_agera_5: str | None = None,
        weather_past: int = 10,  # past time step in weather
        weather_vars: list | None = None,
    ) -> None:
        super().__init__()

        if crop_size > 128 or crop_size % 2 != 0:
            # if we wanted we could support >128, but this is not supported right now
            raise ValueError("crop_size must be <= 128 and even")

        self.site_collection = site_collection
        self.list_id = list_id
        self.crop_size = crop_size
        self.max_len = max_len
        self.dataset_type = dataset_type
        self.fsmap = fsmap
        self.openedzips = None
        self.mod = mod
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
            # self.agera_5_df, self.weather_vars = compute_past_weather(
            #     final_df=agera_5_df,
            #     all_vars=weather_vars,
            #     past_days=self.weather_past,
            # )

            print(f"Loaded weather variables: \n {self.path_agera_5} ")
        else:
            self.agera_5_df = None
            self.weather_vars = None

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index: int) -> S1ForcSITSSample | S2ForcSITSSample:
        if not self.openedzips:
            self.openedzips = self.fsmap.open_all()
        if not self.rand_train:
            self.rand_train = random.Random()

        if self.dataset_type == "train":
            rand = self.rand_train
        else:
            rand = random.Random(index)

        site_id = self.list_id[index]
        site_sits = self.site_collection.get_site(site_id)

        n_max = min(self.max_len, site_sits.n_dates)
        indices_sits = self.time_sampling.sample_single_modality(
            site_sits.acquisitions_metadata, n_max, rand
        )
        roi = get_random_roi(self.crop_size, rand)

        if self.mod == "s2":
            req_s2_10m = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=site_id,
                time_indices=indices_sits,
                bands=S2_BANDS_10M,
                roi=roi,
            )
            req_s2_20m = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=req_s2_10m.site_id,
                time_indices=req_s2_10m.time_indices,
                bands=S2_BANDS_20M,
                roi=(roi[0] // 2, roi[1] // 2, roi[2] // 2, roi[3] // 2),
            )
            req = ziptif.loader.BundleReadRequest(
                subrequests={
                    "S2_10m": req_s2_10m,
                    "S2_20m": req_s2_20m,
                },
            )

        elif self.mod == "s1":
            req_s1 = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=site_id,
                time_indices=indices_sits,
                bands=["VV", "VH"],
                roi=roi,
            )
            req = ziptif.loader.BundleReadRequest(
                subrequests={
                    "S1": req_s1,
                },
            )
        else:
            raise NotImplementedError

        result = ziptif.loader.execute_bundle_read_request(
            req, openedzips=self.openedzips
        )

        if self.mod == "s2":
            s2_sits = s2_timeseries_to_sits(
                result.timeseries["S2_10m"], result.timeseries["S2_20m"]
            )
            if self.agera_5_df is not None:
                days = result.timeseries["S2_10m"].stac_metadata["datetime"]
                aux = extract_agera_5_feat(
                    self.agera_5_df,
                    times=days,
                    patch_id=site_id,
                    weather_col=self.weather_vars,
                    past_days=self.weather_past,
                )
                return S2ForcSITSSample(
                    content=s2_sits.content,
                    time=s2_sits.time,
                    validity_mask=s2_sits.validity_mask,
                    weather_var=aux.to(dtype=s2_sits.content.dtype),
                )
            else:
                return s2_sits
        elif self.mod == "s1":
            if self.agera_5_df is None:
                return s1_timeseries_to_sits(result.timeseries["S1"])
            else:
                days = result.timeseries["S1"].stac_metadata["datetime"]
                s1_sits = s1_timeseries_to_sits(result.timeseries["S1"])
                aux = extract_agera_5_feat(
                    self.agera_5_df,
                    times=days,
                    patch_id=site_id,
                    weather_col=self.weather_vars,
                )

                return S1ForcSITSSample(
                    content=s1_sits.content,
                    time=s1_sits.time,
                    validity_mask=s1_sits.validity_mask,
                    weather_var=aux,
                    angles=s1_sits.angles,
                )

        else:
            raise NotImplementedError


class MonoMTestForecastDataset(MonoMForecastDataset):
    def __init__(
        self,
        site_collection: ziptif.ziptif.SiteCollection,
        list_id: list,
        fsmap: ziptif.loader.FileSystemMapping,
        dataset_type: Literal["train", "val", "test"] = "train",
        max_len: int = 15,
        crop_size: int = 64,
        mod: Literal["s1", "s2"] = "s2",
        time_sampling: TimeSamplingStrategy = RANDOM_SAMPLING_STRATEGY,
        path_agera_5: str | None = None,
        weather_past: int = 10,
        weather_vars: list | None = None,
    ) -> None:
        super().__init__(
            site_collection,
            list_id,
            fsmap,
            dataset_type,
            max_len,
            crop_size,
            mod,
            time_sampling,
            path_agera_5,
            weather_past,
            weather_vars,
        )

    def __getitem__(self, index: int) -> S1ForcSITSSample | S2ForcSITSSample:
        if not self.openedzips:
            self.openedzips = self.fsmap.open_all()
        if not self.rand_train:
            self.rand_train = random.Random()

        if self.dataset_type == "train":
            rand = self.rand_train
        else:
            rand = random.Random(index)

        site_id = self.list_id[index]
        site_sits = self.site_collection.get_site(site_id)

        n_max = min(self.max_len, site_sits.n_dates)

        indices_sits = [i for i in range(site_sits.n_dates)][-n_max:]
        roi = get_random_roi(self.crop_size, rand)

        if self.mod == "s2":
            req_s2_10m = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=site_id,
                time_indices=indices_sits,
                bands=S2_BANDS_10M,
                roi=roi,
            )
            req_s2_20m = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=req_s2_10m.site_id,
                time_indices=req_s2_10m.time_indices,
                bands=S2_BANDS_20M,
                roi=(roi[0] // 2, roi[1] // 2, roi[2] // 2, roi[3] // 2),
            )
            req = ziptif.loader.BundleReadRequest(
                subrequests={
                    "S2_10m": req_s2_10m,
                    "S2_20m": req_s2_20m,
                },
            )

        elif self.mod == "s1":
            req_s1 = ziptif.loader.ReadRequest(
                site_collection=self.site_collection,
                site_id=site_id,
                time_indices=indices_sits,
                bands=["VV", "VH"],
                roi=roi,
            )
            req = ziptif.loader.BundleReadRequest(
                subrequests={
                    "S1": req_s1,
                },
            )
        else:
            raise NotImplementedError

        result = ziptif.loader.execute_bundle_read_request(
            req, openedzips=self.openedzips
        )

        if self.mod == "s2":
            s2_sits = s2_timeseries_to_sits(
                result.timeseries["S2_10m"], result.timeseries["S2_20m"]
            )
            if self.agera_5_df is not None:
                days = result.timeseries["S2_10m"].stac_metadata["datetime"]
                aux = extract_agera_5_feat(
                    self.agera_5_df,
                    times=days,
                    patch_id=site_id,
                    weather_col=self.weather_vars,
                    past_days=self.weather_past,
                )
                return S2ForcSITSSample(
                    content=s2_sits.content,
                    time=s2_sits.time,
                    validity_mask=s2_sits.validity_mask,
                    weather_var=aux.to(dtype=s2_sits.content.dtype),
                )
            else:
                return s2_sits
        elif self.mod == "s1":
            if self.agera_5_df is None:
                return s1_timeseries_to_sits(result.timeseries["S1"])
            else:
                days = result.timeseries["S1"].stac_metadata["datetime"]
                s1_sits = s1_timeseries_to_sits(result.timeseries["S1"])
                aux = extract_agera_5_feat(
                    self.agera_5_df,
                    times=days,
                    patch_id=site_id,
                    weather_col=self.weather_vars,
                )

                return S1ForcSITSSample(
                    content=s1_sits.content,
                    time=s1_sits.time,
                    validity_mask=s1_sits.validity_mask,
                    weather_var=aux,
                    angles=s1_sits.angles,
                )

        else:
            raise NotImplementedError
