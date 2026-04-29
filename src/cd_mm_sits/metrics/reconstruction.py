from typing import Literal

import torch
from torch import Tensor
from torchmetrics import MeanSquaredError, Metric

from cd_mm_sits.lightning_module.supervised_forecast import (
    MonoMOutShared,
    OutSharedStep,
)


class MaskedMSEMetric(Metric):
    def __init__(self, mod: Literal["s2", "s1"], num_outputs: int = 1):
        super().__init__()
        self.metric = MeanSquaredError()
        self.mod = mod

    def update(self, output: OutSharedStep) -> None:
        preds, target = _return_pred_trg_for_one_mod(output=output, mod=self.mod)
        if preds.numel() != 0:
            self.metric.update(preds, target)
        else:
            print("no unmasked values")

    def compute(self) -> Tensor:
        output = self.metric.compute()
        assert ~torch.all(torch.isnan(output))
        return torch.nanmean(output)


def _return_pred_trg_for_one_mod(
    output: OutSharedStep, mod: Literal["s1", "s2"]
) -> tuple[Tensor, Tensor]:
    if isinstance(output, MonoMOutShared):
        preds = output.pred
        target = output.target
        mask = output.mask
    elif mod == "s1":
        preds = output.pred_s1
        target = output.trg_s1
        mask = output.mask_s1
    elif mod == "s2":
        preds = output.pred_s2
        target = output.trg_s2
        mask = output.mask_s2
    else:
        raise NotImplementedError
    return torch.masked_select(preds, mask=mask), torch.masked_select(target, mask=mask)
