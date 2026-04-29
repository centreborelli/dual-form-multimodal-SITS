from random import Random

import polars as pl

from cd_mm_sits.data.datamodule.time_sampling import (
    ContiguousTimeSamplingStrategy,
    RandomTimeSamplingStrategy,
)


def test_random():
    sampler = RandomTimeSamplingStrategy()
    rand = Random(0)
    metadatas = pl.DataFrame(
        {
            "datetime": pl.select(
                pl.date_range(pl.datetime(2020, 1, 1), pl.datetime(2021, 1, 1), "1d")
            )
        }
    )

    time_indices = sampler.sample_single_modality(metadatas, n_max=5, rand=rand)

    assert len(time_indices) == 5
    assert all(0 <= idx < len(metadatas) for idx in time_indices)
    assert sorted(time_indices) == time_indices


def test_contiguous():
    sampler = ContiguousTimeSamplingStrategy()
    rand = Random(0)
    metadatas = pl.DataFrame(
        {
            "datetime": pl.select(
                pl.date_range(pl.datetime(2020, 1, 1), pl.datetime(2021, 1, 1), "1d")
            )
        }
    )

    time_indices = sampler.sample_single_modality(metadatas, n_max=5, rand=rand)

    assert len(time_indices) == 5
    assert all(0 <= idx < len(metadatas) for idx in time_indices)
    assert sorted(time_indices) == time_indices
    # make sure they are contiguous
    assert time_indices == list(range(time_indices[0], time_indices[0] + 5))
