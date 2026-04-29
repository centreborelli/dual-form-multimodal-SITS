"""Callback for mulitmodal supervised forecasting"""

from typing import Literal

from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from cd_mm_sits.callback.utils_display import plot_one_mod
from cd_mm_sits.lightning_module.supervised_forecast import (
    MMSupervisedForecast,
)


class MonoMForecClb(Callback):
    """Callback to visualise forecasting task"""

    def __init__(
        self,
        device="cpu",
        log_every_n_steps: int = 4,
        mod: Literal["s1", "s2"] = "s2",
        bands: list | None | int = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        self.mod = mod
        if (bands is None) and mod == "s2":
            self.bands = [2, 1, 0]
        if (bands is None) and mod == "s1":
            self.bands = 0
        assert self.bands is not None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MMSupervisedForecast,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return
        if outputs is not None:
            fig, ax = plot_one_mod(
                outputs.pred, outputs.target, bands=self.bands, normalize=True
            )

            for logg in trainer.loggers:
                if isinstance(logg, TensorBoardLogger):
                    # Log to TensorBoard
                    trainer.logger.experiment.add_figure(
                        tag=f"S1/Sample_{batch_idx}",
                        figure=fig,
                        global_step=trainer.global_step,
                    )
