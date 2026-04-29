from typing import Literal

import torch.nn
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from cd_mm_sits.data.batch_class import MMForcBatch
from cd_mm_sits.data.datamodule.custome_collate_fn import (
    collate_fn_mm_forc,
    collate_fn_mm_forc_test,
)
from cd_mm_sits.data.datamodule.time_sampling import (
    RANDOM_SAMPLING_STRATEGY,
    TimeSamplingStrategy,
)
from cd_mm_sits.data.datamodule.utils import ZipTifPaths, split_dataset
from cd_mm_sits.data.dataset.mm_forecast import MMForecastDataset, MMTestForecastDataset


class MMForecastDataModule(LightningDataModule):
    def __init__(
        self,
        ziptif_paths: ZipTifPaths,
        batch_size: int,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        train_ratio: float = 0.7,
        max_len: int = 15,
        crop_size: int = 64,
        s2_transform: torch.nn.Module | None = None,
        time_sampling: TimeSamplingStrategy = RANDOM_SAMPLING_STRATEGY,
        path_agera_5: str | None = None,
        weather_past: int = 10,  # past time step in weather
        weather_vars: list | None = None,
        persistent_workers: bool = True,
        consecutive_samples_in_test: bool = True,
    ):
        super().__init__()
        self.ziptif_paths = ziptif_paths
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.max_len = max_len
        self.crop_size = crop_size
        self.s2_transform = s2_transform
        self.time_sampling = time_sampling
        self.path_agera_5 = path_agera_5
        self.weather_vars = weather_vars
        self.weather_past = weather_past
        self.persistent_workers = persistent_workers
        self.consecutive_samples_in_test = consecutive_samples_in_test

    def setup(self, stage: Literal["fit", "test"]):
        site_collection_s1 = self.ziptif_paths.read_site_collection("s1")
        site_collection_s2 = self.ziptif_paths.read_site_collection("s2")
        fsmap = self.ziptif_paths.get_fsmap()

        site_ids_s1 = sorted([sid for sid in site_collection_s1.get_site_ids()])
        site_ids_s2 = sorted([sid for sid in site_collection_s2.get_site_ids()])
        site_ids = sorted(list(set(site_ids_s1) & set(site_ids_s2)))
        train_data, val_data, test_data = split_dataset(site_ids, self.train_ratio)
        print(
            f"MMForecastDatase/n° of samples: train: {len(train_data)} val: {len(val_data)} test: {len(test_data)}"
        )

        self.data_train = MMForecastDataset(
            site_collection_s1=site_collection_s1,
            site_collection_s2=site_collection_s2,
            list_id=train_data,
            fsmap=fsmap,
            dataset_type="train",
            max_len=self.max_len,
            crop_size=self.crop_size,
            time_sampling=self.time_sampling,
            path_agera_5=self.path_agera_5,
            weather_past=self.weather_past,
            weather_vars=self.weather_vars,
        )
        self.data_val = MMForecastDataset(
            site_collection_s1=site_collection_s1,
            site_collection_s2=site_collection_s2,
            list_id=val_data,
            fsmap=fsmap,
            dataset_type="val",
            max_len=self.max_len,
            crop_size=self.crop_size,
            time_sampling=self.time_sampling,
            path_agera_5=self.path_agera_5,
            weather_past=self.weather_past,
            weather_vars=self.weather_vars,
        )
        if self.consecutive_samples_in_test:
            self.data_test = MMTestForecastDataset(
                site_collection_s1=site_collection_s1,
                site_collection_s2=site_collection_s2,
                list_id=test_data,
                fsmap=fsmap,
                dataset_type="test",
                max_len=self.max_len,
                crop_size=self.crop_size,
                time_sampling=self.time_sampling,
                path_agera_5=self.path_agera_5,
                weather_past=self.weather_past,
                weather_vars=self.weather_vars,
            )
        else:
            self.data_test = MMForecastDataset(
                site_collection_s1=site_collection_s1,
                site_collection_s2=site_collection_s2,
                list_id=test_data,
                fsmap=fsmap,
                dataset_type="test",
                max_len=self.max_len,
                crop_size=self.crop_size,
                time_sampling=self.time_sampling,
                path_agera_5=self.path_agera_5,
                weather_past=self.weather_past,
                weather_vars=self.weather_vars,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_forc,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_forc,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,  # mandatory for collate_fn_mm_sp_test
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_forc_test,
            persistent_workers=self.persistent_workers,
        )

    def transfer_batch_to_device(
        self, batch: MMForcBatch, device, dataloader_idx
    ) -> MMForcBatch:
        """
        Mandatory when using personalized data class as output
         of the dataset __getitem__ method
        Parameters
        ----------
        batch :
        device :
        dataloader_idx :

        Returns
        -------

        """
        if isinstance(batch, MMForcBatch):
            # move all tensors in your custom data structure to the device
            return batch.to_device(device)

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def on_after_batch_transfer(
        self, batch: MMForcBatch, dataloader_idx: int
    ) -> MMForcBatch:
        """
        Motivations: apply transform on GPU not on CPU,
         to avoid transfert of float32 from CPU to GPU
        (Tricks from Julien Michel). CPU data are int16.
        Parameters
        ----------
        batch :
        dataloader_idx :

        Returns
        -------

        """
        assert isinstance(batch, MMForcBatch)

        if self.s2_transform is not None:
            sits_s2 = self.s2_transform(batch.s2.content)
            batch.s2.content = sits_s2

        return batch
