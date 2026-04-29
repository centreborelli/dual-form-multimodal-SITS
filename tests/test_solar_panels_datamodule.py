from pathlib import Path

import pytest

from cd_mm_sits.data.batch_class import CDSupBatch
from cd_mm_sits.data.datamodule.solar_panels_datamodule import SPDataModule


@pytest.mark.iris_local
def test_solar_panels_datamodule(datasets_root: Path):
    datamodule = SPDataModule(
        dataset_path="/home/iris/Documents/datasets/mm_solar_panel/zarr_ex",
        path_dir_csv="/home/iris/Documents/datasets/solar_panels_npy",
    )
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    for i in range(2):
        batch = next(iter(train_dataloader))
        assert isinstance(batch, CDSupBatch)
        assert batch.sits is not None


@pytest.mark.iris_local
def test_solar_panels_datamodule_kfold(datasets_root: Path):
    datamodule = SPDataModule(
        dataset_path="/home/iris/Documents/datasets/mm_solar_panel/zarr_ex",
        path_dir_csv="/home/iris/Documents/datasets/solar_panels_npy",
        fold_expe=1,
    )
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    for i in range(2):
        batch = next(iter(train_dataloader))
        assert isinstance(batch, CDSupBatch)
        assert batch.sits is not None
