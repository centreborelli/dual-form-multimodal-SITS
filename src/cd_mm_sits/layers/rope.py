from functools import lru_cache
from typing import override

import torch
import torch.nn as nn

from cd_mm_sits.layers.xpos import (
    _get_theta_xpos,
    fixed_pos_embedding_3d,
    fixed_time_pos_embedding_3d,
)


@staticmethod
@lru_cache(maxsize=128)
def _get_theta(d: int, base: int, device: torch.device, dtype: torch.dtype):
    return 1 / (base ** (torch.arange(0, d, 2, device=device, dtype=dtype) / d))


class MyROPE(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        T_base: float = 1e4,
    ) -> None:
        super().__init__()
        self.T_base = T_base
        head_dim = embed_dim
        self.head_dim = head_dim

    def _load_hp(self) -> dict:
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "base": self.T_base,
            suffix + "d": self.head_dim,
        }

    def forward(self, x: torch.Tensor):
        x_sin = torch.empty(x.shape, device=x.device)
        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D
        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2
        theta_t = fixed_pos_embedding_3d(theta=theta, x_shape=x.shape)  # B,T,D/2
        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D
        return rope


class MyTimeROPE(MyROPE):
    """A modified version of XPOS that changes according to acquisition time"""

    @override
    def forward(self, x, time):
        x_sin = torch.empty(x.shape, device=x.device)

        theta_full = torch.empty(x.shape, device=x.device)  # B,T,D

        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta_xpos(
            self.head_dim, base=self.T_base, device=x.device, dtype=x.dtype
        )  # a vector of size D/2

        theta_t = fixed_time_pos_embedding_3d(theta=theta, time=time)

        theta_full[..., 0::2] = theta_t
        theta_full[..., 1::2] = theta_t

        # theta_full shape is B,T,D
        x_sin[..., 0::2] = -x[..., 1::2]
        x_sin[..., 1::2] = x[..., 0::2]
        rope = torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )  # B,T,D

        return rope
