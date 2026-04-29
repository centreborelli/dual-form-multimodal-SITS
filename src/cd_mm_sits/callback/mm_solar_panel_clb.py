"""File where relevant callback to track solar panel are reported"""

import matplotlib.pyplot as plt
import torch.nn as nn
from lightning import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.trainer import Trainer
from matplotlib.colors import BoundaryNorm, ListedColormap

from cd_mm_sits.data.batch_class import MMSpSupBatch
from cd_mm_sits.lightning_module.mm_supervised_sp_detection import (
    MMSupervisedOut,
    MMSupervisedSP,
)


class PlotMMSP(Callback):
    def __init__(self, num_samples=1, device="cpu", log_every_n_steps: int = 4):
        super().__init__()
        self.num_samples = num_samples
        self.log_every_n_steps = log_every_n_steps
        self.device = device
        self.cmap = ListedColormap(["gray", "black", "red"])
        bounds = [-1.5, -0.5, 0.5, 1.5]  # boundaries between -1, 0, 1
        self.norm = BoundaryNorm(bounds, self.cmap.N)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: MMSupervisedSP,
        outputs: MMSupervisedOut,
        batch: MMSpSupBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if batch_idx % self.log_every_n_steps != 0:
            return
        if outputs is not None:
            pred = nn.functional.sigmoid(outputs.out).cpu()
            label = outputs.label.cpu()
            len_T = pred.shape[1]
            fig, ax = plt.subplots(2, len_T, figsize=(3 * len_T, 6))
            for i in range(len_T):
                ax[0, i].imshow(
                    label[0, i, ...].numpy(),
                    label="Label",
                    norm=self.norm,
                    cmap=self.cmap,
                )
                ax[1, i].imshow(
                    pred[0, i, ...].numpy(),
                    label="Prediction",
                    vmin=0,
                    vmax=1,
                    cmap="seismic",
                )
                fig.tight_layout()
            for logg in trainer.loggers:
                if isinstance(logg, TensorBoardLogger):
                    # Log to TensorBoard
                    trainer.logger.experiment.add_figure(
                        tag=f"TrainPred_vs_Label/Sample_{batch_idx}",
                        figure=fig,
                        global_step=trainer.global_step,
                    )
