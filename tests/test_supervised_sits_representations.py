from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torchmetrics
from hydra import compose, initialize
from hydra.utils import instantiate
from utils import rdn_batch, rdn_defor_batch

from cd_mm_sits.lightning_module.distance import ScaledCosineDistance
from cd_mm_sits.lightning_module.hydra_dataclass import CAWConfig, OptimizerAdamConfig
from cd_mm_sits.lightning_module.losses import FocalLoss
from cd_mm_sits.lightning_module.supervised_sits_representations import (
    SupervisedRepr,
    TrainConfigSupRepr,
)


@pytest.mark.critical
def test_supervised_repr_forward():
    assert Path("./config/supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfigSupRepr(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        distance=nn.PairwiseDistance(p=2),
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = SupervisedRepr(model=model, train_config=train_config)
    t1, t2, c, h, w = 10, 13, 10, 64, 64
    dummy_batch = rdn_batch(t1, t2, c, h, w)
    out, *_ = pl_module.forward(dummy_batch)
    assert out.shape[0] == 2
    assert out.shape[-1] == 1


@pytest.mark.critical
def test_supervised_repr_shared_step():
    assert Path("./config/supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfigSupRepr(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        distance=ScaledCosineDistance(),
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = SupervisedRepr(model=model, train_config=train_config)
    t1, t2, c, h, w = 10, 13, 10, 64, 64
    dummy_batch = rdn_defor_batch(t1, t2, c, h, w)
    output = pl_module.shared_step(dummy_batch)
    assert output.out.shape == (2, h, w)
    assert torch.min(output.out) == 0
    assert torch.max(output.out) == 1
    # Check that parameters got gradients

    # Check that input got gradients
    # assert dummy_batch.sits.grad is not None, "Input did not receive gradients"


@pytest.mark.critical
def test_supervised_repr_shared_step_var_loss():
    assert Path("./config/supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfigSupRepr(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        distance=ScaledCosineDistance(),
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = SupervisedRepr(
        model=model, train_config=train_config, variance_loss=True
    )
    t1, t2, c, h, w = 10, 13, 10, 64, 64
    dummy_batch = rdn_defor_batch(t1, t2, c, h, w)
    dummy_batch.label = torch.zeros(2, h, w)
    output = pl_module.shared_step(dummy_batch)
    assert output.out.shape == (2, h, w)
    assert torch.min(output.out) == 0
    assert torch.max(output.out) == 1
    assert output.loss_var is not None
    assert output.out.requires_grad
