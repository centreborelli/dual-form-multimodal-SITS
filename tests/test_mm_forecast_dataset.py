from pathlib import Path

import torch

from cd_mm_sits.data.dataset.mm_forecast import MMForecastDataset


def get_mm_forecast_dataset(
    datasets_root: Path, max_len: int, crop_size: int
) -> MMForecastDataset:
    from cd_mm_sits.data.datamodule.mm_forecast_datamodule import ZipTifPaths

    ziptif_paths = ZipTifPaths(
        index_paths={
            "s1": [datasets_root.joinpath("mm_forecast/v1-s1.ziptif").as_posix()],
            "s2": [datasets_root.joinpath("mm_forecast/v1-s2.ziptif").as_posix()],
        },
        zip_paths={
            "v1-s1": datasets_root.joinpath("mm_forecast/v1-s1.zip").as_posix(),
            "v1-s2": datasets_root.joinpath("mm_forecast/v1-s2.zip").as_posix(),
        },
    )
    site_collection_s1 = ziptif_paths.read_site_collection("s1")
    site_collection_s2 = ziptif_paths.read_site_collection("s2")
    fsmap = ziptif_paths.get_fsmap()

    site_ids_s1 = sorted([sid for sid in site_collection_s1.get_site_ids()])
    site_ids_s2 = sorted([sid for sid in site_collection_s2.get_site_ids()])
    site_ids = sorted(list(set(site_ids_s1) & set(site_ids_s2)))

    ds = MMForecastDataset(
        site_collection_s1=site_collection_s1,
        site_collection_s2=site_collection_s2,
        list_id=site_ids,
        fsmap=fsmap,
        max_len=max_len,
        crop_size=crop_size,
    )
    return ds


def test_getitem(datasets_root: Path):
    ds = get_mm_forecast_dataset(datasets_root, max_len=5, crop_size=64)

    item = ds[0]

    assert item.s1.content.shape[0] <= 5
    assert item.s1.content.shape[1] == 2
    assert item.s1.content.shape[2] == 64
    assert item.s1.content.shape[3] == 64
    assert item.s1.content.dtype == torch.float32
    assert item.s1.time.shape[0] == item.s1.content.shape[0]
    assert len(item.s1.time.shape) == 1
    assert item.s1.time.dtype == torch.float32
    assert item.s1.validity_mask.shape == (
        item.s1.content.shape[0],
        item.s1.content.shape[2],
        item.s1.content.shape[3],
    )
    assert item.s1.validity_mask.dtype == torch.bool

    assert item.s2.content.shape[0] <= 5
    assert item.s2.content.shape[1] == 10
    assert item.s2.content.shape[2] == 64
    assert item.s2.content.shape[3] == 64
    assert (
        # TODO: at some point, we will want the dataset to return uint16 tensors
        item.s2.content.dtype == torch.float32
    )
    assert item.s2.time.shape[0] == item.s2.content.shape[0]
    assert len(item.s2.time.shape) == 1
    assert item.s2.time.dtype == torch.float32
    assert item.s2.validity_mask.shape == (
        item.s2.content.shape[0],
        item.s2.content.shape[2],
        item.s2.content.shape[3],
    )
    assert item.s2.validity_mask.dtype == torch.bool
