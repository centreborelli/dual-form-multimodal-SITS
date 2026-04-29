"""File which describe the BradS1
datamodule for the deforestation task"""

from pathlib import Path

from torch.utils.data import DataLoader

from cd_mm_sits.data.datamodule.custome_collate_fn import (
    collate_fn_brads1,
    collate_fn_brads1_test,
)
from cd_mm_sits.data.datamodule.template_datamodule import TemplateDataModule
from cd_mm_sits.data.datamodule.utils import load_transform_one_mod, split_dataset
from cd_mm_sits.data.dataset.brads1_defor import BradS1Dataset


class BradS1DataModule(TemplateDataModule):
    def __init__(
        self,
        dataset_path: str,
        path_dir_csv: str,
        dataset_name="dataset",
        max_len_s2: int = 60,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        batch_size: int = 2,
        s2_band: list | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        split_seed: int | None = 1,
    ):
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
            path_dir_csv=self.path_dir_csv, mod="s1"
        )
        self.dict_classes = {"No change": 0, "Change": 1}
        self.labels = list(self.dict_classes.values())
        self.num_classes = len(self.dict_classes.values())
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.split_seed = split_seed

    def setup(self, stage: str) -> None:
        l_files = [path for path in Path(self.dataset_path).rglob("*.pt")]
        train_data, val_data, test_data = split_dataset(
            l_files, self.train_ratio, seed=self.split_seed
        )
        self.data_train = BradS1Dataset(
            path_dataset=self.dataset_path,
            l_files=train_data,
            transform=self.s2_transform.transform,
        )
        self.data_val = BradS1Dataset(
            path_dataset=self.dataset_path,
            l_files=val_data,
            transform=self.s2_transform.transform,
        )
        self.data_test = BradS1Dataset(
            path_dataset=self.dataset_path,
            l_files=test_data,
            transform=self.s2_transform.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_brads1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_brads1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=collate_fn_brads1_test,
        )
