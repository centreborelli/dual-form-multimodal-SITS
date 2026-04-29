"""Callback for mulitmodal supervised forecasting"""

from lightning import Callback
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from cd_mm_sits.callback.utils_display import plot_one_mod
from cd_mm_sits.data.batch_class import MMForcBatch
from cd_mm_sits.lightning_module.supervised_forecast import (
    MMSupervisedForecast,
    OutSharedStep,
)


class MMForecClb(Callback):
    """Callback to visualise forecasting task"""

    def __init__(
        self,
        device="cpu",
        log_every_n_steps: int = 4,
        s1_band: int = 0,
        s2_bands: list | None = None,
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        if s2_bands is None:
            s2_bands = [2, 1, 0]
        self.s2_bands = s2_bands
        self.s1_band = s1_band

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MMSupervisedForecast,
        outputs: OutSharedStep,
        batch: MMForcBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return
        if outputs is not None:
            fig_s1, ax_s1 = plot_one_mod(
                outputs.pred_s1, outputs.trg_s1, bands=self.s1_band
            )
            fig_s2, ax_s2 = plot_one_mod(
                outputs.pred_s2, outputs.trg_s2, bands=self.s2_bands, normalize=True
            )
            for logg in trainer.loggers:
                if isinstance(logg, TensorBoardLogger):
                    # Log to TensorBoard
                    trainer.logger.experiment.add_figure(
                        tag=f"S1/Sample_{batch_idx}",
                        figure=fig_s1,
                        global_step=trainer.global_step,
                    )
                    trainer.logger.experiment.add_figure(
                        tag=f"S2/Sample_{batch_idx}",
                        figure=fig_s2,
                        global_step=trainer.global_step,
                    )
