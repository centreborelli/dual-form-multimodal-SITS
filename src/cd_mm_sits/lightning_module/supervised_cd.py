"""File which contains the lightning module relevant
for supervised change detection on S2 SITS"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from cd_mm_sits.constant.label import IGNORE_LABEL
from cd_mm_sits.data.batch_class import CDSupBatch
from cd_mm_sits.lightning_module.template_module import TemplateModule, TrainConfig
from cd_mm_sits.model.basic_baselines import OnlyUnet
from cd_mm_sits.model.sste import SSTE


@dataclass
class SupervisedCDOut:
    loss: Tensor
    out_masked: Tensor
    label_masked: Tensor
    out: Tensor
    label: Tensor


class SupervisedCD(TemplateModule):
    def __init__(
        self, train_config: TrainConfig, model: SSTE | OnlyUnet, compile: bool = False
    ) -> None:
        super().__init__(
            train_config,
        )
        if compile:
            self.model = torch.compile(model)
        else:
            self.model = model
        metrics = train_config.metrics  # metrics should be scalar metrics !!!
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        dict_hp = {"lr": train_config.lr, "batch_size": train_config.batch_size}
        dict_hp.update(model.save_hp)
        self.save_hyperparameters(dict_hp)

    def forward(
        self,
        batch: CDSupBatch,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Forward function for a supervised change detection

        :param batch:
        :param need_weights:
        :param is_causal: wether the attention is computed
        in a auto-regressive
        way (means can't attend to future prediction)
        :returns:
        - a Tensor of shape B,maxT,H,W,C with max T the maximum temporal
        length of the SITS within the batch and
        C the number of class (here usually 2)
        - A list of the attention weights in the temporal transformer or
        a list of None if need_weights set to False
        """
        out, weights, _ = self.model(
            batch=batch.sits,
            time=batch.time,
            need_weights=need_weights,
        )
        return out, weights

    def shared_step(self, batch: CDSupBatch) -> SupervisedCDOut:
        out, _ = self.forward(batch)

        if batch.label.is_nested:
            label = torch.nested.to_padded_tensor(batch.label, padding=IGNORE_LABEL)
        else:
            label = batch.label
        if batch.mask_site is not None:
            label = torch.masked_fill(
                label, ~batch.mask_site[:, None, ...].bool(), IGNORE_LABEL
            )
        mask_label = label == IGNORE_LABEL

        out_masked = out[..., 0][
            ~mask_label
        ]  # kind of dirty fix, expect 1 layer as output
        if out_masked.nelement() == 0:
            return None

        loss = self.loss(
            out_masked,
            label[~mask_label],
        )  # here we suppose that the loss takes
        return SupervisedCDOut(
            loss=loss,
            out_masked=nn.functional.sigmoid(out_masked),
            label_masked=label[~mask_label].long(),
            out=out,
            label=label,
        )

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(self.hparams, {"val_loss": 0, "val_f1_score": 0})
        return super().on_train_start()

    def training_step(self, batch: CDSupBatch, batch_idx: int) -> Tensor:
        output = self.shared_step(batch)
        if output is None:
            return None
        self.train_metrics.update((output.out_masked > 0.5).int(), output.label_masked)
        self.log(
            "train_loss",
            output.loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        return output.loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.compute()
        self.log_dict(
            self.train_metrics,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def validation_step(self, batch: CDSupBatch, batch_idx: int):
        output = self.shared_step(batch)
        if output is None:
            return None
        self.log(
            "val_loss",
            output.loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
            prog_bar=True,
        )
        self.val_metrics.update((output.out_masked > 0.5).int(), output.label_masked)
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self.val_metrics.compute()

        self.log_dict(
            outputs,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def test_step(self, batch: CDSupBatch, batch_idx: int):
        output = self.shared_step(batch)
        if output is None:
            return None
        self.test_metrics.update((output.out_masked > 0.5).int(), output.label_masked)
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
