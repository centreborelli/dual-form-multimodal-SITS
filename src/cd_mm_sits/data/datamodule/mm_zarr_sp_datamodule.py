from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cd_mm_sits.data.batch_class import MMSpSupBatch
from cd_mm_sits.data.datamodule.custome_collate_fn import (
    collate_fn_mm_sp,
    collate_fn_mm_sp_test,
)
from cd_mm_sits.data.datamodule.template_datamodule import TemplateDataModule
from cd_mm_sits.data.datamodule.utils import (
    load_transform_one_mod,
    read_txt_fold,
    split_dataset,
)
from cd_mm_sits.data.dataset.sp_mm_zarr import MMMaxLen, MMTransform, ZarrMMSolarPanel


class MMSPDataModule(TemplateDataModule):
    """
    LightningDataModule class dedicated to
    the Solar Panel data
    """

    def __init__(
        self,
        dataset_path: str,
        path_dir_csv: str,
        dataset_name="dataset",
        max_len_s2: int = 30,
        max_len_s1: int = 30,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        batch_size: int = 1,
        s2_band: list | None = None,
        crop_size: int = 64,
        fold_expe: int | None = None,
        split_seed: int | None = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        is_mask_site: bool = True,
    ):
        """

        :param dataset_path: path to the directory containing all .npy
        :param path_dir_csv: path to the directory containing the dataset_s2.csv
        :param dataset_name:
        :param max_len_s2: the maximum temporal length
        :param num_workers:
        :param prefetch_factor:
        :param batch_size:
        :param s2_band:
        :param crop_size: maximum spatial size
        :param kfold_expe: if not None, look at the k-fold file
        :param split_seed: only if kfold_expe=None
        :param train_ratio:only if kfold_expe=None
        :param val_ratio: only if kfold_expe=None
        :param test_ratio: only if kfold_expe=None
        :returns:

        """

        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            path_dir_csv=path_dir_csv,
            max_len_s2=max_len_s2,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            s2_band=s2_band,
        )
        self.s2_transform = load_transform_one_mod(
            path_dir_csv=self.path_dir_csv, mod="s2_batch"
        )
        self.s1_transform = load_transform_one_mod(
            path_dir_csv=self.path_dir_csv, mod="s1"
        )

        self.crop_size = crop_size
        self.dict_classes = {"No change": 0, "Change": 1}
        self.labels = list(self.dict_classes.values())
        self.num_classes = len(self.dict_classes.values())
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed
        self.fold_expe = fold_expe
        self.max_len_s1 = max_len_s1
        self.is_mask_site = is_mask_site

    def setup(self, stage: Any):
        if self.fold_expe is None:
            with open(Path(self.dataset_path).joinpath("sites.txt")) as f:
                txt = f.read()
            self.list_id = txt.split("\n")[:-1]
            train_data, val_data, test_data = split_dataset(
                self.list_id, self.train_ratio
            )
        else:
            train_data, val_data, test_data = read_txt_fold(
                self.fold_expe, dataset_path=self.dataset_path
            )
        self.data_train = ZarrMMSolarPanel(
            path_dataset=self.dataset_path,
            list_id=train_data,
            transform=MMTransform(None, None),
            dataset_type="train",
            max_len=MMMaxLen(self.max_len_s2, self.max_len_s1),
            crop_size=self.crop_size,
            is_mask_site=self.is_mask_site,
        )
        self.data_val = ZarrMMSolarPanel(
            path_dataset=self.dataset_path,
            list_id=val_data,
            transform=MMTransform(None, None),
            dataset_type="val",
            max_len=MMMaxLen(self.max_len_s2, self.max_len_s1),
            crop_size=self.crop_size,
            is_mask_site=self.is_mask_site,
        )

        self.data_test = ZarrMMSolarPanel(
            path_dataset=self.dataset_path,
            list_id=test_data,
            transform=MMTransform(None, None),
            dataset_type="test",
            max_len=MMMaxLen(self.max_len_s2, self.max_len_s1),
            crop_size=self.crop_size,
            is_mask_site=self.is_mask_site,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_sp,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_sp,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,  # ELSE CAUSING MEMORY ISSUE
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_sp_test,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,  # ELSE CAUSING MEMORY ISSUE
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_mm_sp_test,
        )

    def transfer_batch_to_device(
        self, batch: MMSpSupBatch, device: torch.device, dataloader_idx: int
    ) -> MMSpSupBatch:
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
        return batch.to_device(device)

    def on_after_batch_transfer(
        self, batch: MMSpSupBatch, dataloader_idx: int
    ) -> MMSpSupBatch:
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
        if self.s2_transform.transform is not None:
            sits_s2 = self.s2_transform.transform(batch.s2.sits)

            batch.s2.sits = sits_s2

        return batch
