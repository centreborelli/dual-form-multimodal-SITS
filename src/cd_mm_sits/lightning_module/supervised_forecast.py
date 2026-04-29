from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from einops import rearrange

from cd_mm_sits.constant.model import S1_IDX, S2_IDX
from cd_mm_sits.data.batch_class import MMForcBatch, SITSBatch
from cd_mm_sits.lightning_module.template_module import TemplateModule, TrainConfig
from cd_mm_sits.model.forecaster import (
    InputForecForward,
    MMShallowForecast,
    RopeMMShallowForecaster,
)
from cd_mm_sits.model.mm_sste import MMSSTE
from cd_mm_sits.model.sste import SSTE
from cd_mm_sits.model.utils import cat_nested_along_time


@dataclass
class AuxVar:
    angles: torch.Tensor | None
    weather: torch.Tensor | None


@dataclass
class OutForward:
    last_repr: torch.Tensor
    pred_s1: torch.Tensor
    pred_s2: torch.Tensor
    weights: None | torch.Tensor
    sorted_idx: None | torch.Tensor
    idx_s1_pred: torch.Tensor
    idx_s2_pred: torch.Tensor
    idx_s1_trg: torch.Tensor
    idx_s2_trg: torch.Tensor

    def print_shape(self):
        print(
            f"pred_s1 {self.pred_s1.shape}\n pred_s2 {self.pred_s2.shape}"
            "\n target s1 {self.idx_s1_trg.shape} \n"
            "target s2 {self.idx_s2_trg.shape}\n"
        )


@dataclass
class OutSharedStep:
    pred_s1: torch.Tensor
    pred_s2: torch.Tensor
    trg_s1: torch.Tensor
    trg_s2: torch.Tensor
    mask_s1: torch.Tensor
    mask_s2: torch.Tensor


@dataclass
class MMLoss:
    loss_s1: torch.Tensor
    loss_s2: torch.Tensor


class MMSupervisedForecast(TemplateModule):
    """Lightning Module for a supervised multi-modal forecasting task"""

    def __init__(
        self,
        train_config: TrainConfig,
        model: MMSSTE,
        shallow_forecaster: MMShallowForecast | RopeMMShallowForecaster,
        compile: bool = False,
        pred_after_n: int = 10,
        w_s2: float = 1,
        use_delta_times: bool = False,
    ) -> None:
        super().__init__(train_config)
        if compile:
            self.model = torch.compile(model)  # torch.compile(model)
        else:
            self.model = model
        self.shallow_forecaster = shallow_forecaster
        assert isinstance(shallow_forecaster, nn.Module)
        if train_config.metrics is not None:
            metrics = train_config.metrics  # metrics should abe scalar metrics !!!
            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None
        self.pred_after_n = pred_after_n  # forecast is done for each lat repre
        self.w_s2 = w_s2
        if isinstance(self.shallow_forecaster, RopeMMShallowForecaster):
            use_delta_times = True
        self.use_delta_times = use_delta_times
        # after pred_after_n acquisitions
        dict_hp = {
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "pred_after_n": pred_after_n,
            "use_delta_times": self.use_delta_times,
            "w_s2": w_s2,
        }
        dict_hp.update(model.save_hp)
        dict_hp.update(self.shallow_forecaster._load_hp())
        self.save_hyperparameters(dict_hp)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {"val_loss": torch.tensor(0.0)},
        )
        return super().on_train_start

    def forward(self, batch: MMForcBatch, need_weights: bool = False) -> OutForward:
        """Given a multimodal batch process with the multimodal encoder
        s1 and s2 sits (spectral content+time for each)
        :param batch: the input
        :param need_weights: if True return attention weights
        :returns:

        """
        # Step 1: Process the MM SITS with the MM encoder
        out_forward = self.model(
            sits_s1=batch.s1.content,
            sits_s2=batch.s2.content,
            time_s1=batch.s1.time,
            time_s2=batch.s2.time,
            s1_orbites=batch.s1.orbites,
            need_weights=need_weights,
        )
        lat_repr = out_forward.output

        B, T, H, W, C = lat_repr.shape
        # Step 2: Extract the latent representations implied in forecast
        last_repr = rearrange(
            lat_repr[:, self.pred_after_n : -1, ...], " B T H W C -> (B T ) H W C "
        )  # all repr after
        # print(out_forward.sorted_mod.shape)

        # Step 3: Get the modality of the next element to predict
        # do not forget temporal shift

        time_pred_s1, idx_s1_pred = self._extract_time_forecast(
            self.pred_after_n + 1,
            T + 1,
            time=out_forward.sorted_times,
            mod=out_forward.sorted_mod,
            B=B,
            H=last_repr.shape[1],
            W=last_repr.shape[2],
            mod_value=S1_IDX,
        )
        time_pred_s2, idx_s2_pred = self._extract_time_forecast(
            self.pred_after_n + 1,
            T + 1,
            time=out_forward.sorted_times,
            mod=out_forward.sorted_mod,
            B=B,
            H=last_repr.shape[1],
            W=last_repr.shape[2],
            mod_value=S2_IDX,
        )
        time_input_s1, _ = self._extract_time_forecast(
            self.pred_after_n,
            T,
            time=out_forward.sorted_times,
            mod=out_forward.sorted_mod,
            B=B,
            H=last_repr.shape[1],
            W=last_repr.shape[2],
            mod_value=S1_IDX,
        )
        time_input_s2, _ = self._extract_time_forecast(
            self.pred_after_n,
            T,
            time=out_forward.sorted_times,
            mod=out_forward.sorted_mod,
            B=B,
            H=last_repr.shape[1],
            W=last_repr.shape[2],
            mod_value=S2_IDX,
        )
        mod = rearrange(
            out_forward.sorted_mod[:, self.pred_after_n + 1 :, ...],
            "(B H W)  T -> (B T)  H W",
            B=B,
            H=last_repr.shape[1],
            W=last_repr.shape[2],
        )

        sorted_idx_pixel = rearrange(
            out_forward.sorted_idx, "(B H W ) T -> B T H W ", H=H, W=W
        )[..., 0, 0]
        idx_s1_trg, idx_s2_trg = _find_correct_global_monom_idx(
            batch=batch,
            sorted_idx=sorted_idx_pixel,
            mod=mod[..., 0, 0],
            pred_after_n=self.pred_after_n,
        )

        # we get the index in B T tensor of all s1 data.
        # we need to insert false on those who are in interval

        # print(time_pred_s1.shape)
        # Step 6: Extract the lat_repr used to predict each mod.
        repr_for_s1 = last_repr[idx_s1_pred, ...]
        repr_for_s2 = last_repr[idx_s2_pred, ...]
        aux_vars_s1 = _process_aux_var(batch.s1, idx_s1_trg)
        aux_vars_s2 = _process_aux_var(batch.s2, idx_s2_trg)
        input_s1 = InputForecForward(
            lat_repr=repr_for_s1,
            target_time=time_pred_s1,
            mod="s1",
            input_time=time_input_s1,
            weather=aux_vars_s1.weather,
            angles=aux_vars_s1.angles,
        )

        input_s2 = InputForecForward(
            lat_repr=repr_for_s2,
            target_time=time_pred_s2,
            mod="s2",
            input_time=time_input_s2,
            weather=aux_vars_s2.weather,  # TODO later add S2 angles
        )
        out_s1 = self.shallow_forecaster(input_s1)  # can decode either s1 or s2
        # Step 3b: Predict from the last latent representation the target of modality s2

        out_s2 = self.shallow_forecaster(input_s2)

        return OutForward(
            last_repr=last_repr,
            pred_s1=out_s1,
            pred_s2=out_s2,
            weights=out_forward.weights,
            sorted_idx=out_forward.sorted_idx,
            idx_s1_pred=idx_s1_pred,
            idx_s2_pred=idx_s2_pred,
            idx_s1_trg=idx_s1_trg,
            idx_s2_trg=idx_s2_trg,
        )

    def _extract_time_forecast(
        self,
        n_start: int,
        n_end: int,
        time: torch.Tensor,
        mod: torch.Tensor,  # (BT) H W
        mod_value: int,
        B: int,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """extract time information for input or target.
        If use delta_times pred and target have the same time"""
        if self.use_delta_times:
            times = torch.diff(time, dim=1)[:, self.pred_after_n :, ...]

            times = rearrange(
                times,
                "(B H W ) T  -> (B T) H W",
                B=B,
                H=H,
                W=W,
            )
            mod = rearrange(
                mod[..., self.pred_after_n + 1 :], "(B H W) T -> (B T) H W", H=H, W=W
            )
        else:
            times = rearrange(
                time[:, n_start:n_end, ...],
                "(B H W) T -> (B T) H W",
                B=B,
                H=H,
                W=W,
            )
            mod = rearrange(mod[..., n_start:n_end], "(B H W) T -> (B T) H W", H=H, W=W)

        idx = torch.where(mod[..., 0, 0] == mod_value)[
            0
        ]  # pad_mask is 0, means value not padded

        return times[idx, ..., 0, 0], idx

    def shared_step(self, batch: MMForcBatch) -> tuple[OutSharedStep, MMLoss]:
        """Process the input and compute the loss.
        :param batch:
        :returns:

        """
        # b = batch.s1.time.shape[0]
        out = self.forward(batch)

        # out.print_shape()

        if batch.s1.content.is_nested:
            target_s1 = batch.s1.content.values()
            mask_s1 = batch.s1.validity_mask.values()
            mask_s2 = batch.s2.validity_mask.values()
            target_s2 = batch.s2.content.values()
        else:
            target_s1 = rearrange(batch.s1.content, "B T C H W -> (B T) C H W ")
            target_s2 = rearrange(batch.s2.content, "B T C H W -> (B T) C H W ")
            mask_s2 = rearrange(batch.s2.validity_mask, "B T H W -> (B T)  H W ")
            mask_s1 = rearrange(batch.s1.validity_mask, " B T H W -> (B T ) H W ")
        trg_s1 = target_s1[out.idx_s1_trg, ...]
        trg_s2 = target_s2[out.idx_s2_trg, ...]
        mask_s1 = mask_s1[out.idx_s1_trg, ...].unsqueeze_(1).bool()
        mask_s2 = mask_s2[out.idx_s2_trg, ...].unsqueeze_(1).bool()
        loss_s1 = self.loss(
            torch.masked_select(out.pred_s1, mask_s1),
            torch.masked_select(trg_s1, mask_s1),
        )
        loss_s2 = self.loss(
            torch.masked_select(out.pred_s2, mask_s2),
            torch.masked_select(trg_s2, mask_s2),
        )

        return (
            OutSharedStep(
                pred_s1=out.pred_s1,
                trg_s1=trg_s1,
                pred_s2=out.pred_s2,
                trg_s2=trg_s2,
                mask_s1=mask_s1,
                mask_s2=mask_s2,
            ),
            MMLoss(loss_s1=loss_s1, loss_s2=loss_s2),
        )

    def training_step(self, batch: MMForcBatch, batch_idx: int) -> torch.Tensor | None:
        output, mm_loss = self.shared_step(batch)
        self.log(
            "train_loss_s1",
            mm_loss.loss_s1,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        self.log(
            "train_loss_s2",
            mm_loss.loss_s2,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        if torch.isnan(mm_loss.loss_s2):
            loss = mm_loss.loss_s1
        elif torch.isnan(mm_loss.loss_s1):
            loss = mm_loss.loss_s2
        loss = mm_loss.loss_s1 + self.w_s2 * mm_loss.loss_s2
        self.log("train_loss", loss, on_epoch=True, on_step=False, batch_size=self.bs)
        return loss

    def validation_step(self, batch: MMForcBatch, batch_idx: int) -> OutSharedStep:
        output, mm_loss = self.shared_step(batch)
        self.log(
            "val_loss_s1",
            mm_loss.loss_s1,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        self.log(
            "val_loss_s2",
            mm_loss.loss_s2,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        if torch.isnan(mm_loss.loss_s2):
            loss = mm_loss.loss_s1
        elif torch.isnan(mm_loss.loss_s1):
            loss = mm_loss.loss_s2
        else:
            loss = mm_loss.loss_s1 + self.w_s2 * mm_loss.loss_s2
        self.log("val_loss", loss, on_epoch=True, on_step=False, batch_size=self.bs)
        return output

    def test_step(self, batch: MMForcBatch, batch_idx: int) -> OutSharedStep:
        output, mm_loss = self.shared_step(batch)
        self.log(
            "test_loss_s1",
            mm_loss.loss_s1,
            on_epoch=False,
            on_step=True,
            batch_size=self.bs,
        )
        self.log(
            "test_loss_s2",
            mm_loss.loss_s2,
            on_epoch=False,
            on_step=True,
            batch_size=self.bs,
        )
        if torch.isnan(mm_loss.loss_s2) and ~torch.isnan(mm_loss.loss_s1):
            loss = mm_loss.loss_s1
        elif torch.isnan(mm_loss.loss_s1) and ~torch.isnan(mm_loss.loss_s2):
            loss = self.w_s2 * mm_loss.loss_s2
        elif ~torch.isnan(mm_loss.loss_s2) and ~torch.isnan(mm_loss.loss_s2):
            loss = mm_loss.loss_s1 + self.w_s2 * mm_loss.loss_s2
        else:
            return None

        self.log("test_loss", loss, on_epoch=True, on_step=True, batch_size=self.bs)
        if self.test_metrics is not None:
            self.test_metrics.update(output)
        return output


@dataclass
class MonoMOutShared:
    pred: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor
    mask: torch.Tensor


class MonoMForecast(TemplateModule):
    """Lightning Module for mono-modal forecasting"""

    def __init__(
        self,
        train_config: TrainConfig,
        model: SSTE,
        shallow_forecaster: MMShallowForecast | RopeMMShallowForecaster,
        compile: bool = False,
        pred_after_n: int = 10,
        use_delta_times: bool = False,
        mod: Literal["s1", "s2"] = "s2",
    ) -> None:
        super().__init__(train_config)
        if compile:
            self.model = torch.compile(model)  # torch.compile(model)
        else:
            self.model = model
        self.shallow_forecaster = shallow_forecaster
        assert isinstance(shallow_forecaster, nn.Module)
        if train_config.metrics is not None:
            metrics = train_config.metrics  # metrics should abe scalar metrics !!!
            self.train_metrics = metrics.clone(prefix="train_")
            self.val_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

        self.pred_after_n = pred_after_n  # forecast is done for each lat repre
        if isinstance(self.shallow_forecaster, RopeMMShallowForecaster):
            use_delta_times = True
        self.use_delta_times = use_delta_times
        self.mod = mod
        # after pred_after_n acquisitions
        dict_hp = {
            "lr": train_config.lr,
            "batch_size": train_config.batch_size,
            "pred_after_n": pred_after_n,
            "use_delta_times": self.use_delta_times,
            "mod": self.mod,
        }
        dict_hp.update(model.save_hp)
        dict_hp.update(self.shallow_forecaster._load_hp())
        self.save_hyperparameters(dict_hp)

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams,
            {"val_loss": torch.tensor(0.0)},
        )
        return super().on_train_start

    def forward(
        self, batch: SITSBatch, need_weights: bool = False, right_product: bool = False
    ) -> Any:
        lat_repr, weights, pad_mask = self.model(
            batch=batch.content,
            time=batch.time,
            need_weights=need_weights,
            right_product=right_product,
        )
        B, T, H, W, C = lat_repr.shape
        # print(f"lat_repr {lat_repr.shape}")
        last_repr = rearrange(
            lat_repr[:, self.pred_after_n : -1, ...], "B T H W C -> (B T ) H W C"
        )

        time = _process_var(batch.time)
        assert time is not None
        time = time.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        time_target, idx = self._extract_time_forecast(
            self.pred_after_n + 1, T + 1, time=time, pad_mask=pad_mask, B=B, H=H, W=W
        )
        time_input, _ = self._extract_time_forecast(
            self.pred_after_n, T, time=time, pad_mask=pad_mask, B=B, H=H, W=W
        )
        aux_vars = _process_aux_var(batch, idx)
        input_fore = InputForecForward(
            target_time=time_target,
            lat_repr=last_repr[idx, ...],
            mod=self.mod,
            input_time=time_input,
            weather=aux_vars.weather,
            angles=aux_vars.angles,
        )
        out = self.shallow_forecaster(input_fore)  # can decode either s1 or s2
        return last_repr, out, weights, idx

    def shared_step(self, batch: SITSBatch):
        last_repr, pred, weights, idx = self.forward(batch)
        BT, C, H, W = last_repr.shape
        if batch.content.is_nested:
            target = torch.nested.to_padded_tensor(batch.content, padding=0)
            mask = torch.nested.to_padded_tensor(batch.validity_mask, padding=0)
        else:
            target = batch.content
            mask = batch.validity_mask
        target = target[:, self.pred_after_n + 1 :, ...]
        mask = mask[:, self.pred_after_n + 1 :, ...]
        target = rearrange(target, "B T C H W -> (B T ) C H W ")
        mask = rearrange(mask, " B T H W -> (B T ) H W")

        target = target[idx, ...]
        mask = mask[idx, ...].unsqueeze_(1).bool()
        loss = self.loss(
            torch.masked_select(pred, mask=mask), torch.masked_select(target, mask=mask)
        )
        return MonoMOutShared(pred=pred, target=target, loss=loss, mask=mask)

    def training_step(self, batch: SITSBatch, batch_idx: int) -> torch.Tensor | None:
        out_shared = self.shared_step(batch=batch)
        if torch.isnan(out_shared.loss):
            return None
        self.log(
            "train_loss",
            out_shared.loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        return out_shared.loss

    def validation_step(
        self, batch: SITSBatch, batch_idx: int
    ) -> MonoMOutShared | None:
        out_shared = self.shared_step(batch=batch)
        if torch.isnan(out_shared.loss):
            return None
        self.log(
            "val_loss",
            out_shared.loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        return out_shared

    def test_step(self, batch: SITSBatch, batch_idx: int) -> MonoMOutShared | None:
        out_shared = self.shared_step(batch=batch)
        if torch.isnan(out_shared.loss):
            return None
        self.log(
            "test_loss",
            out_shared.loss,
            on_epoch=True,
            on_step=False,
            batch_size=self.bs,
        )
        if self.test_metrics is not None:
            self.test_metrics.update(out_shared)
        return out_shared

    def _extract_time_forecast(
        self,
        n_start: int,
        n_end: int,
        time: torch.Tensor,
        pad_mask: torch.Tensor,
        B: int,
        H: int,
        W: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """extract time information for input or target.
        If use delta_times pred and target have the same time"""
        if self.use_delta_times:
            times = torch.diff(time, dim=1)[:, self.pred_after_n :, ...]
            times = rearrange(
                times,
                "B T H W -> (B T) H W",
                B=B,
                H=H,
                W=W,
            )
            pad_mask = rearrange(
                pad_mask[..., self.pred_after_n + 1 :],
                "(B H W) T -> (B T) H W",
                H=H,
                W=W,
            )
        else:
            times = rearrange(
                time[:, n_start:n_end, ...],
                "B T H W -> (B T) H W",
                B=B,
                H=H,
                W=W,
            )
            pad_mask = rearrange(
                pad_mask[..., n_start:n_end], "(B H W) T -> (B T) H W", H=H, W=W
            )

        idx = torch.where(pad_mask[..., 0, 0] == 0)[0]

        # pad_mask is 0, means value not padded

        return times[idx, ..., 0, 0], idx


def split_mm_pred(
    sorted_mm_sits, sorted_mod: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    idx_s1 = torch.where(sorted_mod == S1_IDX)[0]
    idx_s2 = torch.where(sorted_mod == S2_IDX)[0]

    s1_sits_pred = sorted_mm_sits[idx_s1, ...]
    s2_sits_pred = sorted_mm_sits[idx_s2, ...]
    return s1_sits_pred, s2_sits_pred


def _process_var(var: torch.Tensor | None):
    if var is not None:
        if var.is_nested:
            return torch.nested.to_padded_tensor(var, padding=0)
        else:
            return var

    return None


def _find_correct_global_monom_idx(
    batch: MMForcBatch,
    sorted_idx: torch.Tensor,
    mod: torch.Tensor,
    pred_after_n: int,
):
    """Get the global mono-modal idx !"""
    # step 1 extract the global monom idx

    s1_global_idx = batch.extract_s1_idx()  # global idx in b*t1 tensor
    s2_global_idx = batch.extract_s2_idx()  # global idx in b*t2
    # step 2 reproduce the mm tensor fusion
    mm_fusion = cat_nested_along_time(x_1=s1_global_idx, x_2=s2_global_idx)  # B T
    # TODO padd if nested
    if batch.s1.content.is_nested:
        mm_fusion = torch.nested.to_padded_tensor(
            mm_fusion, padding=0
        )  # B,T,n_mod_feat
    sorted_idx = sorted_idx.to(device=mm_fusion.device)
    sorted_mm_idx = rearrange(
        torch.gather(mm_fusion, 1, index=sorted_idx)[:, pred_after_n + 1 :],
        "B T -> ( B T )",
    )  # sort by acquisition dates
    # step 3: split per mode (we are in the monom b*t space)
    # print(sorted_mm_idx.shape)
    # print(mod.shape)

    sorted_mm_idx_s1 = sorted_mm_idx[mod.to(sorted_mm_idx.device) == S1_IDX]
    sorted_mm_idx_s2 = sorted_mm_idx[mod.to(sorted_mm_idx.device) == S2_IDX]
    return sorted_mm_idx_s1, sorted_mm_idx_s2


def _process_aux_var(batch: SITSBatch, idx) -> AuxVar:
    weather_var = _process_var(batch.weather_var)
    angles_var = _process_var(batch.angles)

    if weather_var is not None:
        weather_var = rearrange(weather_var, " B T Tbef D -> (B T) Tbef D")[idx, ...]
    if angles_var is not None:
        angles_var = rearrange(angles_var, " B T D -> (B T ) D")[idx, ...]
    return AuxVar(weather=weather_var, angles=angles_var)
