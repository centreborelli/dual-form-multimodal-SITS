"""Class which serve as foundation to all other lightning module class"""

from dataclasses import dataclass
from typing import Any

import torch
import torchmetrics
from hydra.utils import instantiate
from lightning import LightningModule
from torch.nn.modules.loss import _WeightedLoss

from cd_mm_sits.lightning_module.hydra_dataclass import (
    CAWConfig,
)


@dataclass
class TrainConfig:
    loss: _WeightedLoss
    batch_size: int
    optimizer: Any  # OptimizerAdamConfig | OptimizerAdamWConfig
    optimizer_monitor: str
    scheduler: CAWConfig
    lr: float
    metrics: torchmetrics.MetricCollection | None


class TemplateModule(LightningModule):
    """
    Template for all lightning Module implemented
    """

    def __init__(self, train_config: TrainConfig):
        super().__init__()
        self.train_config = train_config
        self.learning_rate = train_config.lr
        # self.save_hyperparameters(ignore=["train_config", "datamodule"])
        self.scheduler_config = train_config.scheduler
        self.optimizer_config = train_config.optimizer
        self.bs = train_config.batch_size
        self.loss = train_config.loss

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = instantiate(
            self.optimizer_config, params=self.parameters(), lr=self.learning_rate
        )
        sch: torch.optim.lr_scheduler = instantiate(
            self.scheduler_config, optimizer=optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.train_config.optimizer_monitor,
                "strict": False,
            },
        }
