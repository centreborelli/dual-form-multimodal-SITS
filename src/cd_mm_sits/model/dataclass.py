from dataclasses import dataclass

from torch import Tensor


@dataclass
class State:
    kv_t: Tensor | None = None
    k_cum_t: Tensor | None = None
    step: int = 1

    def __post_init__(self):
        if self.kv_t is None:
            assert self.k_cum_t is None
        if self.k_cum_t is None:
            assert self.kv_t is None
