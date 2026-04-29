from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from cd_mm_sits.constant.label import IGNORE_LABEL
from cd_mm_sits.data.batch_class import MMSpSupBatch
from cd_mm_sits.lightning_module.template_module import TemplateModule, TrainConfig
from cd_mm_sits.model.mm_sste import MMSSTE
from cd_mm_sits.model.utils import cat_nested_along_time


@dataclass
class MMSupervisedOut:
    loss: Tensor
    out_masked: Tensor
    label_masked: Tensor
    out: Tensor
    label: Tensor


class MMSupervisedSP(TemplateModule):
    def __init__(
        self, train_config: TrainConfig, model: MMSSTE, compile: bool = False
    ) -> None:
        super().__init__(train_config)
        if compile:
            self.model = torch.compile(model)  # torch.compile(model)
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
        self, batch: MMSpSupBatch, need_weights: bool = False
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        # Construct the input TS with either S1,S2 or S1+S2 TS.
        out_forward = self.model(
            sits_s1=batch.s1.sits,
            sits_s2=batch.s2.sits,
            time_s1=batch.s1.time,
            time_s2=batch.s2.time,
            s1_orbites=batch.s1.orbites,
            need_weights=need_weights,
        )
        return out_forward.output, out_forward.sorted_idx, out_forward.weights

    def shared_step(self, batch: MMSpSupBatch) -> MMSupervisedOut | None:
        mm_labels = cat_nested_along_time(batch.s1.label, batch.s2.label)  # B,j,h,w
        # add temporal padding
        if mm_labels.is_nested:
            mm_labels = torch.nested.to_padded_tensor(
                mm_labels, padding=IGNORE_LABEL
            )  # B,T,h,w

        out, sorted_idx, _ = self.forward(batch)
        # merge label along the ragged time dimension

        # when pixel are oustide the spatial mask of the site, consider masking
        # mask_labels = mask_labels.masked_fill_(
        #     ~batch.mask_site[:, None, ...].bool(), True
        # )

        if batch.mask_site is not None:
            mm_labels = torch.masked_fill(
                mm_labels, ~batch.mask_site[:, None, ...].bool(), IGNORE_LABEL
            )
        sorted_idx = rearrange(
            sorted_idx,
            "(B H W) T ->  B T H W",
            H=mm_labels.shape[-2],
            W=mm_labels.shape[-1],
        )

        # print(mm_labels.shape, sorted_idx.shape)
        mm_labels = torch.gather(mm_labels, 1, sorted_idx)  # MANDATORY, reorder labels
        # mask for the labels account for padding and non existing labels
        mask_labels = mm_labels == IGNORE_LABEL
        masked_labels = mm_labels[~mask_labels]
        if masked_labels.nelement() == 0:
            return None

        masked_pred = out[..., 0][~mask_labels]

        loss = self.loss(masked_pred, masked_labels)
        return MMSupervisedOut(
            loss=loss,
            out_masked=nn.functional.sigmoid(masked_pred),
            label_masked=masked_labels,
            out=out,
            label=mm_labels,
        )

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {"val_loss": torch.tensor(0.0), "val_f1_score": torch.tensor(0.0)},
        )
        return super().on_train_start()

    def training_step(self, batch: MMSpSupBatch, batch_idx: int) -> Tensor | None:
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
        self.train_metrics.reset()

    def validation_step(self, batch: MMSpSupBatch, batch_idx: int):
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
        self.val_metrics.reset()

    def test_step(self, batch: MMSpSupBatch, batch_idx: int):
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
