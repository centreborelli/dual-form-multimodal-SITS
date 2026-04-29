from dataclasses import dataclass

import torch
import torchvision
from einops import rearrange
from torch import Tensor


@dataclass
class Stats:
    """
    Store stats
    """

    median: list
    qmin: list
    qmax: list


@dataclass
class OneTransform:
    """
    Store transform and its associated stats
    """

    transform: torch.nn.Module | None = None
    stats: Stats | None = None


class Clip(torch.nn.Module):
    """
    Clip transform for data, clip each band between two values qmin and qmax
    """

    def __init__(self, qmin: list, qmax: list, s2_partial: bool = False):
        super().__init__()

        self.qmin = rearrange(
            torch.Tensor(qmin), "c -> c 1 1 1"
        )  # reshape so that it is broadcastable
        self.qmax = rearrange(torch.Tensor(qmax), "c ->  c  1 1 1 ")
        self.s2_partial = s2_partial

    def forward(self, tensor: Tensor) -> Tensor:
        tmp_tensor = tensor
        assert len(tensor.shape) == 4
        qmax = self.qmax.to(tmp_tensor)
        qmin = self.qmin.to(tmp_tensor)
        tmp_tensor[torch.isnan(tmp_tensor)] = torch.max(qmax) + 100
        tmp_tensor = torch.min(
            torch.max(tmp_tensor, qmin), qmax
        )  # clip values on the quantile

        return tmp_tensor

    def __repr__(self):
        return self.__class__.__name__ + f"( qmin={self.qmin} , qmax={self.qmax})"


class S2Normalize(torch.nn.Module):
    """Robust Normalization"""

    def __init__(
        self,
        med: list | tuple,
        scale: list | tuple,
    ):
        super().__init__()
        self.med = med  # reshape so that it is broadcastable
        self.scale = scale
        self.transform = torchvision.transforms.Normalize(mean=med, std=scale)

    def forward(self, tensor: Tensor):
        tensor = rearrange(tensor, "c t h w -> t c h w").to(torch.float)
        tensor = self.transform(tensor)
        return rearrange(tensor, "t c h w -> c t h w")

    def __repr__(self):
        return self.__class__.__name__ + f"( med={self.med} , scale={self.scale})"


class S2BatchNormalize(torch.nn.Module):
    """Robust Normalization"""

    def __init__(
        self,
        med: list | tuple,
        scale: list | tuple,
    ):
        super().__init__()
        self.med = med  # reshape so that it is broadcastable
        self.scale = scale
        self.transform = torchvision.transforms.Normalize(mean=med, std=scale)

    def forward(self, tensor: Tensor):
        med = torch.tensor(self.med).view(1, 1, -1, 1, 1).to(tensor.device)
        scale = torch.tensor(self.scale).view(1, 1, -1, 1, 1).to(tensor.device)
        return (tensor - med) / scale

    def __repr__(self):
        return self.__class__.__name__ + f"( med={self.med} , scale={self.scale})"


class SqrtTransform(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sqrt(x)
