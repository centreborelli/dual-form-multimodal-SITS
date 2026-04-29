from __future__ import annotations

import abc
import random
from typing import override

import polars as pl


class TimeSamplingStrategy(abc.ABC):
    @abc.abstractmethod
    def sample(
        self,
        # modality -> dataframe (containing a `datetime` column)
        metadatas: dict[str, pl.DataFrame],
        n_max: int,
        rand: random.Random,
    ) -> dict[str, list[int]]:
        pass

    def sample_single_modality(
        self,
        metadatas: pl.DataFrame,
        n_max: int,
        rand: random.Random,
    ) -> list[int]:
        return self.sample({"_": metadatas}, n_max, rand)["_"]


class RandomTimeSamplingStrategy(TimeSamplingStrategy):
    @override
    def sample(
        self,
        metadatas: dict[str, pl.DataFrame],
        n_max: int,
        rand: random.Random,
    ) -> dict[str, list[int]]:
        indices = {}
        for mod in metadatas:
            site_sits = metadatas[mod]
            n_dates = len(site_sits)
            idx = sorted(rand.sample(range(n_dates), n_max))
            indices[mod] = idx
        return indices


class ContiguousTimeSamplingStrategy(TimeSamplingStrategy):
    @override
    def sample(
        self,
        metadatas: dict[str, pl.DataFrame],
        n_max: int,
        rand: random.Random,
    ) -> dict[str, list[int]]:
        assert len(metadatas) == 1, (
            f"ContiguousTimeSamplingStrategy only supports single modality, got {list(metadatas.keys())}"
        )
        # TODO: when considering multiple modalities, we need to find a common contiguous range
        indices = {}
        for mod in metadatas:
            site_sits = metadatas[mod]
            n_dates = len(site_sits)
            start_idx = rand.randint(0, n_dates - n_max)
            idx = list(range(start_idx, start_idx + n_max))
            indices[mod] = idx
        return indices


# this instance is used as default value for parameters in some __init__
RANDOM_SAMPLING_STRATEGY = RandomTimeSamplingStrategy()
