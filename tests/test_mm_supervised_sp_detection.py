from pathlib import Path

import pytest
import torch
import torchmetrics
from hydra import compose, initialize
from hydra.utils import instantiate

from cd_mm_sits.data.batch_class import CDSupBatch, MMSpSupBatch
from cd_mm_sits.data.datamodule.mm_zarr_sp_datamodule import MMSPDataModule
from cd_mm_sits.lightning_module.hydra_dataclass import CAWConfig, OptimizerAdamConfig
from cd_mm_sits.lightning_module.losses import FocalLoss
from cd_mm_sits.lightning_module.mm_supervised_sp_detection import MMSupervisedSP
from cd_mm_sits.lightning_module.template_module import TrainConfig


def create_rdn_notnested_batch_mm(
    t11: int = 8,
    t12: int = 20,
    t21: int = 20,
    t22: int = 10,
    c_s1: int = 2,
    c_s2: int = 10,
    h: int = 64,
    w: int = 64,
):
    T1 = max(t11, t12)
    T2 = max(t21, t22)
    batch_1 = torch.randn(2, T1, c_s1, h, w)
    doy_1 = torch.zeros(2, T1)
    label_1 = torch.randn(2, T1, h, w)
    batch_2 = torch.randn(2, T2, c_s2, h, w)
    label_2 = torch.randn(2, T2, h, w)
    doy_2 = torch.randn(2, T2)

    b_s1 = CDSupBatch(sits=batch_1, time=doy_1, label=label_1, orbites=doy_1)
    b_s2 = CDSupBatch(sits=batch_2, time=doy_2, label=label_2)
    return MMSpSupBatch(s1=b_s1, s2=b_s2, mask_site=torch.ones(2, h, w).int())


def create_rdn_batch_mm(
    t11: int = 8,
    t12: int = 20,
    t21: int = 20,
    t22: int = 10,
    c_s1: int = 2,
    c_s2: int = 10,
    h: int = 64,
    w: int = 64,
):
    batch_1 = torch.nested.nested_tensor(
        [torch.randn(t11, c_s1, h, w), torch.randn(t12, c_s1, h, w)],
        layout=torch.jagged,
    )
    label_1 = torch.nested.nested_tensor(
        [torch.randn(t11, h, w), torch.randn(t12, h, w)],
        layout=torch.jagged,
    )
    batch_2 = torch.nested.nested_tensor(
        [torch.randn(t21, c_s2, h, w), torch.randn(t22, c_s2, h, w)],
        layout=torch.jagged,
    )
    label_2 = torch.nested.nested_tensor(
        [torch.randn(t21, h, w), torch.randn(t22, h, w)],
        layout=torch.jagged,
    )
    doy_1 = torch.nested.nested_tensor(
        [torch.arange(t11), torch.arange(t12)], layout=torch.jagged
    )
    doy_2 = torch.nested.nested_tensor(
        [torch.arange(t21) + 1, torch.arange(t22) + 1], layout=torch.jagged
    )

    b_s1 = CDSupBatch(sits=batch_1, time=doy_1, label=label_1, orbites=doy_1)
    b_s2 = CDSupBatch(sits=batch_2, time=doy_2, label=label_2)
    return MMSpSupBatch(s1=b_s1, s2=b_s2, mask_site=torch.ones(2, h, w).int())


@pytest.mark.iris_local
def test_forward():
    assert Path("./config/mm_supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="mm_supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = MMSupervisedSP(model=model, train_config=train_config)
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    batch = create_rdn_batch_mm(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    assert isinstance(pl_module, MMSupervisedSP)
    output, idx, weight = pl_module.forward(batch)
    T = max(t11 + t21, t12 + t22)
    assert output.shape == (2, T, h, w, 1)


@pytest.mark.critical
def test_shared_step():
    assert Path("./config/mm_supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="mm_supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = MMSupervisedSP(model=model, train_config=train_config)
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    batch = create_rdn_batch_mm(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    assert isinstance(pl_module, MMSupervisedSP)
    output = pl_module.shared_step(batch)
    T = max(t11 + t21, t12 + t22)
    assert output.out.shape == (2, T, h, w, 1)
    loss = output.loss
    loss.backward()
    assert loss.requires_grad


@pytest.mark.critical
def test_shared_step_nonested():
    assert Path("./config/mm_supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="mm_supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = MMSupervisedSP(model=model, train_config=train_config)
    t11, t12 = 10, 13
    t21, t22 = 5, 7
    c_s1, c_s2, h, w = 2, 10, 64, 64
    batch = create_rdn_notnested_batch_mm(
        t11=t11, t12=t12, t21=t21, t22=t22, c_s1=c_s1, c_s2=c_s2, h=h, w=w
    )
    assert isinstance(pl_module, MMSupervisedSP)
    output = pl_module.shared_step(batch)
    T = batch.s1.sits.shape[1] + batch.s2.sits.shape[1]
    assert output.out.shape == (2, T, h, w, 1)


@pytest.mark.iris_local
def test_shared_step_with_true_data():
    assert Path("./config/mm_supervised_cd.yaml").exists()
    with initialize(
        config_path="../config/", job_name="test_cd_supervised", version_base="1.2"
    ):
        config = compose(config_name="mm_supervised_cd.yaml")

    model = instantiate(config.model)

    train_config = TrainConfig(
        loss=FocalLoss(),
        batch_size=2,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        metrics=torchmetrics.MetricCollection(
            {"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=2)}
        ),
    )
    pl_module = MMSupervisedSP(model=model, train_config=train_config, compile=False)

    datamodule: MMSPDataModule = instantiate(
        config.datamodule, fold_expe=None, max_len_s2=4, max_len_s1=2, batch_size=1
    )
    datamodule.setup("train")
    train_dataloader = datamodule.val_dataloader()
    iter_dl = iter(train_dataloader)
    batch = next(iter_dl)
    output = pl_module.shared_step(batch)
    lengths_s2 = [s2.shape[0] for s2 in batch.s2.sits]
    lengths_s1 = [s1.shape[0] for s1 in batch.s1.sits]
    T = max([l_1 + l_2 for l_1, l_2 in zip(lengths_s2, lengths_s1, strict=False)])
    assert output.out.shape == (
        datamodule.batch_size,
        T,
        datamodule.crop_size,
        datamodule.crop_size,
        1,
    )
