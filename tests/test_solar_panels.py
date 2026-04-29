from pathlib import Path

from cd_mm_sits.data.datamodule.utils import load_transform_one_mod
from cd_mm_sits.data.dataset.solar_panels import SolarPanel, SPSampleSITS


def test_getitem_solar_panels(datasets_root: Path):
    path_dataset = datasets_root.joinpath("solar_panels_npy")
    with open(path_dataset.joinpath("sites.txt")) as f:
        txt = f.read()
    list_id = txt.split("\n")[:-1]

    s2_transform = load_transform_one_mod(
        path_dir_csv=path_dataset.as_posix(), mod="s2"
    )
    ds = SolarPanel(
        path_dataset=path_dataset.as_posix(),
        transform=s2_transform.transform,
        list_id=list_id,
    )
    item = ds.__getitem__(1)
    assert isinstance(item, SPSampleSITS)
    assert item.sits.shape[1] == 10
    assert item.sits.shape[0] == item.time.shape[0]
    assert item.sits.shape[0] == item.label.shape[0]
    assert item.label.shape[-1] == item.sits.shape[-1]
