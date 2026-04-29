from pathlib import Path

import pytest

from cd_mm_sits.data.datamodule.custome_collate_fn import (
    collate_fn_brads1,
    collate_fn_mm_forc,
    collate_fn_mm_sp,
    collate_fn_sp,
)
from cd_mm_sits.data.datamodule.utils import load_transform_one_mod
from cd_mm_sits.data.dataset.brads1_defor import BradS1Dataset
from cd_mm_sits.data.dataset.solar_panels import SolarPanel
from cd_mm_sits.data.dataset.sp_mm_zarr import MMMaxLen, ZarrMMSolarPanel


def test_collate_fn_sp(datasets_root: Path):
    path_dataset = datasets_root.joinpath("solar_panels_npy")
    s2_transform = load_transform_one_mod(
        path_dir_csv=path_dataset.as_posix(), mod="s2"
    )
    with open(path_dataset.joinpath("sites.txt")) as f:
        txt = f.read()
    list_id = txt.split("\n")[:-1]

    ds = SolarPanel(
        path_dataset=path_dataset.as_posix(),
        transform=s2_transform.transform,
        list_id=list_id,
    )

    item1 = ds.__getitem__(0)
    item2 = ds.__getitem__(2)
    item3 = ds.__getitem__(4)
    batch = collate_fn_sp([item1, item2, item3])
    assert len(batch.sits.unbind()) == 3
    assert len(batch.time.unbind()) == 3
    assert len(batch.label.unbind()) == 3


def test_custom_collate_brads1(datasets_root: Path):
    ds = BradS1Dataset(
        path_dataset=datasets_root.joinpath("BraDD-S1TS_zenodo/Samples").as_posix(),
    )
    item1 = ds.__getitem__(0)
    item2 = ds.__getitem__(2)
    item3 = ds.__getitem__(4)
    batch = collate_fn_brads1([item1, item2, item3])
    assert len(batch.sits.unbind()) == 3
    assert len(batch.time.unbind()) == 3
    assert len(batch.label.unbind()) == 3


@pytest.mark.iris_local
def test_collate_fn_mm_sp():
    path_dataset = "/home/iris/Documents/datasets/mm_solar_panel/zarr_ex"
    zarr_ids = ["7a0xfoy6593ocz4nfoojuaipu7m6ercs"] * 10
    ds = ZarrMMSolarPanel(
        path_dataset=path_dataset,
        list_id=zarr_ids,
        crop_size=128,
        dataset_type="val",
        max_len=MMMaxLen(s2=5, s1=10),
    )
    item1 = ds.__getitem__(0)
    item2 = ds.__getitem__(2)
    item3 = ds.__getitem__(4)
    batch = collate_fn_mm_sp([item1, item2, item3])
    assert batch.s2.sits.shape[0] == 3
    assert batch.s1.sits.shape[0] == 3
    assert batch.mask_site.shape[0] == 3
    assert batch.mask_site.shape[-1] == batch.s2.label.shape[-1]


def test_custom_collate_mm_forc(datasets_root: Path):
    from test_mm_forecast_dataset import get_mm_forecast_dataset

    ds = get_mm_forecast_dataset(datasets_root, max_len=50, crop_size=64)

    item1 = ds.__getitem__(0)
    item2 = ds.__getitem__(2)
    item3 = ds.__getitem__(4)
    batch = collate_fn_mm_forc([item1, item2, item3])
    assert len(batch.s1.content.unbind()) == 3
    assert len(batch.s1.time.unbind()) == 3
    assert len(batch.s1.content.unbind()) == 3
    assert len(batch.s2.time.unbind()) == 3
    assert len(batch.s2.time.unbind()) == 3
    assert len(batch.s2.time.unbind()) == 3
