from pathlib import Path

import pytest
import torch
import torchmetrics
from hydra import compose, initialize
from hydra.utils import instantiate
from utils import rdn_batch

from cd_mm_sits.lightning_module.hydra_dataclass import CAWConfig, OptimizerAdamConfig
from cd_mm_sits.lightning_module.supervised_cd import SupervisedCD
from cd_mm_sits.lightning_module.template_module import TrainConfig


def test_forward(datasets_root: Path):
    assert Path("./config/supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="supervised_cd.yaml", overrides=["server=tests"])

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=torch.nn.BCELoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = SupervisedCD(model=model, train_config=train_config)
    t1, t2, c, h, w = 10, 13, 10, 64, 64
    dummy_batch = rdn_batch(t1, t2, c, h, w)
    out, attn_weights = pl_module.forward(dummy_batch)
    assert out.shape == (2, max(t1, t2), h, w, 1)


@pytest.mark.iris_local
def test_forward_true_data():
    assert Path("./config/supervised_cd.yaml").exists()

    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=torch.nn.BCELoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = SupervisedCD(model=model, train_config=train_config)
    h, w = 64, 64
    datamodule = instantiate(
        config.datamodule, max_len_s2=13, batch_size=2, crop_size=64
    )
    datamodule.setup("train")
    train_dl = iter(datamodule.train_dataloader())
    dummy_batch = next(train_dl)
    out, attn_weights = pl_module.forward(dummy_batch)
    assert out.shape == (2, 13, h, w, 1)
