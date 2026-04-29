import torch
import torch.nn as nn
from torch import Tensor


class ScaledCosineDistance(nn.Module):
    """An implementation of the cosine distance
    which is scaled between 0 and 1
    """

    def __init__(self, dim: int = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.similarity = nn.CosineSimilarity(dim=dim)  # output value between [-1,1]

    def forward(self, x: Tensor, y: Tensor):
        """When the similarity is high (i.e 1) the distance
        should be 0 and 1 when low simiraty (-1)"""

        return (1 - self.similarity(x, y)) / 2


class ClippedMSE(nn.Module):
    def __init__(self, scale: float = 4, p: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.distance = nn.PairwiseDistance(p)
        self.scale = scale

    def forward(self, x: Tensor, y: Tensor):
        return torch.clamp(self.distance(x, y) / self.scale, 0.0, 1.0)
