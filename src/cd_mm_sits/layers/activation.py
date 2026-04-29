import torch.nn as nn
from torch import Tensor


class ModifiedELU(nn.Module):
    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x: Tensor) -> Tensor:
        return 1 + self.elu(x)
