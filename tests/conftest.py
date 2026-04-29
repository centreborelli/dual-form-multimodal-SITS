import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def skip_if_no_datasets():
    """
    $ rsync -avP jeanzay:/lustre/fsn1/projects/rech/jmo/ubc65oh/datasets/ ../tests-datasets
    (or a symlink to another directory)
    """

    root = Path("../tests-datasets")
    if not root.exists():
        pytest.skip(
            "No datasets found, skipping tests that require datasets. "
            "Please download datasets and place them in '../tests-datasets/'. "
            "See `./tests/conftest.py`"
        )

    if "SKIP_DATASET_TESTS" in os.environ and os.environ["SKIP_DATASET_TESTS"] == "1":
        pytest.skip("Skipping tests that require datasets.")

    # for now, we assume that all datasets are present
    datasets = [
        "solar_panels_npy",
        # "mm_solar_panel/zarr_ex",
        "BraDD-S1TS_zenodo/Samples",
    ]
    for dataset in datasets:
        if not root.joinpath(dataset).exists():
            pytest.fail(f"Dataset '{dataset}' missing.")


@pytest.fixture(scope="session")
def datasets_root(skip_if_no_datasets) -> Path:
    return Path("../tests-datasets")
