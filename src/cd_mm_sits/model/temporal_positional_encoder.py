"""
Contains all class for temporal positional encoding
"""

from functools import lru_cache

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoder(nn.Module):
    """
    Traditional Positional encoding as defined by Vaswani 2017.
    Implementation inspired by
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """

    def __init__(self, d: int, T: int = 1000, offset: int = 0):
        """

        Parameters
        ----------
        d : Number of features
        T : The scaling constant
        offset :
        """
        super().__init__()
        self.d = d
        self.T = T
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def _load_hp(self):
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "T": self.T,
        }

    def forward(self, batch_positions: Tensor) -> Tensor:
        """

        Parameters
        ----------    assert False
        batch_positions : (B,T) 2D tensor containing positions.

        Returns
        -------
        a tensor of size B,T,C with C corresponding to self.d
        """
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        return sinusoid_table

    def forward_1d(self, batch_positions: Tensor) -> Tensor:
        """

        Parameters
        ----------    assert False
        batch_positions : (B*T) 1D tensor containing positions.

        Returns
        -------
        a tensor of size B*T,C with C corresponding to self.d
        """
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = batch_positions[:, None] / self.denom[None, :]  # BT x T
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return sinusoid_table


@staticmethod
@lru_cache(maxsize=128)
def _get_theta(d: int, base: int, device: torch.device, dtype: torch.dtype):
    return 1 / (base ** (torch.arange(0, d, 2, device=device, dtype=dtype) / d))


class TimeDeltaRotaryPE(nn.Module):
    """Rotate a tensor using rotary PE, rotation depends of a time delta"""

    def __init__(self, base: int, d: int) -> None:
        super().__init__()
        self.base = base
        self.d = d

    def _load_hp(self) -> dict:
        suffix = "PE/"
        return {
            suffix + "_target_": type(self).__name__,
            suffix + "base": self.base,
            suffix + "d": self.d,
        }

    def forward(self, x: Tensor, delta: Tensor):
        """Rotate a vector x based on delta

        :param x:  B,d
        :param delta: B
        :returns:

        """
        x_sin = torch.empty(x.shape, device=x.device)
        theta_full = torch.empty(x.shape, device=x.device)
        # step value of the rotation for d/2 features (d/2 rotation matrix are employed)
        theta = _get_theta(self.d, base=self.base, device=x.device, dtype=x.dtype)
        theta = torch.einsum("b,d->bd", delta, theta)  # shape B,D
        theta_full[:, 0::2] = theta
        theta_full[:, 1::2] = theta
        # theta_full shape is B,D
        x_sin[:, 0::2] = -x[:, 1::2]
        x_sin[:, 1::2] = x[:, 0::2]
        return torch.mul(x, torch.cos(theta_full)) + torch.mul(
            x_sin, torch.sin(theta_full)
        )
