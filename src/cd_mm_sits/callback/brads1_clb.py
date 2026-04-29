import matplotlib.pyplot as plt
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.trainer import Trainer

from cd_mm_sits.data.batch_class import DeforBatch
from cd_mm_sits.lightning_module.supervised_sits_representations import (
    SupervisedRepr,
    SupervisedReprOut,
)


class PlotChangeMapCallback(Callback):
    def __init__(self, log_every_n_steps: int = 30) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: SupervisedRepr,
        outputs: SupervisedReprOut,
        batch: DeforBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx % self.log_every_n_steps != 0:
            return True
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(outputs.out[0, ...].cpu().numpy(), vmin=0, vmax=1, cmap="seismic")
        ax[0].set_title("Distance maps")
        ax[1].imshow(batch.label[0, ...].cpu().numpy(), vmin=0, vmax=1, cmap="seismic")
        ax[1].set_title("Label")
        fig.tight_layout()
        for logg in trainer.loggers:
            if isinstance(logg, TensorBoardLogger):
                # Log to TensorBoard
                trainer.logger.experiment.add_figure(
                    tag=f"Pred_vs_Label/Sample_{batch_idx}",
                    figure=fig,
                    global_step=trainer.global_step,
                )
