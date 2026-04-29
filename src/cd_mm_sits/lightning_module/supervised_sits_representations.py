"""File which describe the lightning module for
supervised learning of invariant representations"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat
from lightning.pytorch.loggers import Logger
from torch.nn.functional import normalize

from cd_mm_sits.data.batch_class import DeforBatch
from cd_mm_sits.lightning_module.template_module import TemplateModule, TrainConfig
from cd_mm_sits.model.decoder import ConvUpSample
from cd_mm_sits.model.sste import SSTE


@dataclass
class TrainConfigSupRepr(TrainConfig):
    distance: nn.Module


@dataclass
class SupervisedReprOut:
    loss: torch.Tensor
    out: torch.Tensor
    label: torch.Tensor
    loss_var: None | torch.Tensor = None


class SupervisedRepr(TemplateModule):
    def __init__(
        self,
        train_config: TrainConfigSupRepr,
        model: SSTE,
        compile: bool = False,
        variance_loss: bool = False,  # if set to True minimize the variance on batch
        # with no change
        w_var: float = 1,
    ) -> None:
        super().__init__(train_config)
        if compile:
            self.model = torch.compile(model)
        else:
            self.model = model
        metrics = train_config.metrics  # metrics should be scalar metrics !!!
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        dict_hp = {
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "variance_loss": variance_loss,
            "w_var": w_var,
        }
        dict_hp.update({"distance": type(train_config.distance).__name__})
        dict_hp.update(model.save_hp)
        self.distance = train_config.distance
        self.save_hyperparameters(dict_hp)
        self.variance_loss = variance_loss
        self.w_var = w_var

    def forward(
        self, batch: DeforBatch, need_weights: bool = False, **kwargs: Any
    ) -> Any:
        out, weights, key_padding_mask = self.model(
            batch.sits, time=batch.time, need_weights=need_weights
        )
        out = normalize(out, dim=-1)  # normalize across "pseudo-spectral" features

        return out, weights, key_padding_mask

    def compute_distance(self, repr_1: torch.Tensor, repr_2: torch.Tensor):
        b, h, w, c = repr_1.shape
        dis = self.distance(
            rearrange(repr_1, "B H W C -> (B H W) C"),
            rearrange(repr_2, "B H W C -> (B H W) C"),
        )
        return rearrange(dis, "(B H W) -> B H W", B=b, H=h, W=w)

    def shared_step(self, batch: DeforBatch) -> SupervisedReprOut:
        out, weights, key_padding_mask = self.forward(batch)  # B,T,H,W,C

        # padded_out = torch.nested.to_padded_tensor(out, padding=0.0)

        emb_bef = out[batch.batch_idx, batch.idx_bef, ...]
        emb_after = out[batch.batch_idx, batch.idx_after, ...]
        pred = self.compute_distance(emb_bef, emb_after)

        loss = self.loss(pred, batch.label.float())  # the prediction loss
        if self.variance_loss:
            if torch.sum(batch.label.float() > 0) == 0:
                upscale_factor = (
                    2 if isinstance(self.model.last_layer, ConvUpSample) else 1
                )
                key_padding_mask = rearrange(
                    key_padding_mask,
                    "(B H W) T-> B H W T",
                    B=out.shape[0],
                    H=out.shape[2] // upscale_factor,
                )
                key_padding_mask = repeat(
                    key_padding_mask,
                    "B H W T -> B (H r1) (W r2) T",
                    r1=upscale_factor,
                    r2=upscale_factor,
                )
                key_padding_mask = rearrange(key_padding_mask, "B H W T ->(B H W) T")
                key_padding_mask = key_padding_mask.unsqueeze(-1)  # (B H W) T 1
                out = rearrange(out, "B T H W C -> (B H W) T C")

                mask = ~key_padding_mask  # valid positions
                out_masked = out * mask  # zero out padded positions

                # compute mean only on valid values
                mean = out_masked.sum(dim=1, keepdim=True) / mask.sum(
                    dim=1, keepdim=True
                ).clamp(min=1)
                var = ((out_masked - mean) ** 2) * mask
                var_loss = var.sum() / mask.sum().clamp(min=1) + 1e-6
            else:
                var_loss = None
        else:
            var_loss = None
        return SupervisedReprOut(
            loss=loss, out=pred, label=batch.label, loss_var=var_loss
        )

    def on_train_start(self) -> None:
        assert isinstance(self.logger, Logger)
        self.logger.log_hyperparams({"val_loss": 0, "val_f1_score": 0})

        return super().on_train_start()

    def training_step(self, batch: DeforBatch, batch_idx: int) -> torch.Tensor:
        output = self.shared_step(batch)
        self.train_metrics.update(output.out, output.label)
        self.log(
            "train_loss",
            output.loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        if output.loss_var is not None:
            self.log(
                "train_loss_var",
                output.loss_var,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
            return output.loss + self.w_var * output.loss_var
        else:
            return output.loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.compute()
        self.log_dict(
            self.train_metrics,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def validation_step(self, batch: DeforBatch, batch_idx: int):
        output = self.shared_step(batch)
        self.log(
            "val_loss",
            output.loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
            prog_bar=True,
        )
        self.val_metrics.update(output.out, output.label)
        if output.loss_var is not None:
            self.log(
                "val_loss_var",
                output.loss_var,
                on_epoch=True,
                on_step=True,
                batch_size=self.bs,
                prog_bar=True,
            )
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self.val_metrics.compute()
        self.log_dict(
            outputs,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def test_step(self, batch: DeforBatch, batch_idx: int):
        output = self.shared_step(batch)
        self.test_metrics.update(output.out, output.label)
        return output

    def on_test_epoch_end(self) -> None:
        outputs = self.test_metrics.compute()
        outputs = {k: v.to(device="cpu", non_blocking=True) for k, v in outputs.items()}
        self.log_dict(
            outputs,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        self.save_test_metrics = outputs
