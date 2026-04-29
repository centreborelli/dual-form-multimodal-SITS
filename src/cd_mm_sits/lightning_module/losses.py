"""File for custom losses"""

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


class ShiftInvariantLoss(nn.Module):
    """
    Computes the minimum loss over all possible shifts up to max_shift

    Warning: because it crops a subwindow from the input and target
             the network might not learn to predict the borders well.
    """

    def __init__(self, inner: nn.Module, max_shift: int):
        super().__init__()
        self.inner = inner
        self.max_shift = max_shift

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        height, width = input.shape[-2:]
        # inner crop
        ch = height - 2 * self.max_shift
        cw = width - 2 * self.max_shift

        min_loss = None
        for sy in range(-self.max_shift, self.max_shift + 1):
            for sx in range(-self.max_shift, self.max_shift + 1):
                # crop a fixed window from the target
                t = target[
                    ...,
                    self.max_shift : self.max_shift + ch,
                    self.max_shift : self.max_shift + cw,
                ]
                # crop a shifted window from the input
                i = input[
                    ...,
                    self.max_shift + sy : self.max_shift + sy + ch,
                    self.max_shift + sx : self.max_shift + sx + cw,
                ]
                loss = self.inner(i, t)
                if min_loss is None or loss < min_loss:
                    min_loss = loss

        assert min_loss is not None
        return min_loss


class Sar2SarLoss(nn.Module):
    """
    From SAR2SAR: A semi-supervised despeckling algorithm for SAR image
        Dalsasso, E., Denis, L., & Tupin, F. (2021). SAR2SAR: A semi-supervised despeckling algorithm for SAR images. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 14, 4321-4329.
        https://arxiv.org/pdf/2006.15037

    Warning: In our datasets, the SAR images are in amplitude.
             In the paper, they are in log-intensity.
    """

    def __init__(
        self,
        input_unit: Literal["amplitude", "intensity", "log-intensity"],
        reduction: Literal["mean", "sum", "none"] = "mean",
    ):
        super().__init__()
        assert input_unit in ("amplitude", "intensity", "log-intensity")
        self.input_unit = input_unit
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.input_unit == "amplitude" or self.input_unit == "intensity":
            if torch.is_anomaly_check_nan_enabled():
                # verify that there are no negative values in the target, that would indicate an issue in the dataset / dataloader
                assert torch.all(target >= 0), "target contains negative values"

        if self.input_unit == "amplitude":
            input = input**2
            target = target**2

        if self.input_unit != "log-intensity":
            input = torch.log(input + 1e-8)
            target = torch.log(target + 1e-8)

        diff = input - target
        loss = diff + torch.exp(-diff)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        raise ValueError(f"Unknown reduction: {self.reduction}")
