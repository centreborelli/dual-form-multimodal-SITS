import re
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MetricCollection

from cd_mm_sits.data.datamodule.mm_forecast_datamodule import MMForecastDataModule
from cd_mm_sits.data.datamodule.mm_zarr_sp_datamodule import MMSPDataModule
from cd_mm_sits.lightning_module.mm_supervised_sp_detection import MMSupervisedSP
from cd_mm_sits.lightning_module.supervised_cd import SupervisedCD
from cd_mm_sits.lightning_module.supervised_forecast import MMSupervisedForecast
from cd_mm_sits.lightning_module.supervised_sits_representations import SupervisedRepr


def remove_compiled_prefix(state_dict):
    """Remove '_orig_mod.' prefix from state dict keys added by torch.compile()."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    return new_state_dict


def get_best_checkpoint(
    checkpoint_dir: Path,
    metric: str = "val_f1_score",
    mode: Literal["min", "max"] = "max",
):
    assert checkpoint_dir.exists()
    # Pattern to match the val_loss from filenames
    pattern = re.compile(metric + r"=([0-9.]+)\.ckpt")
    best_ckpt = None
    if mode == "min":
        best_val_loss = float("inf")
    else:
        best_val_loss = 0

    if (metric == "best") or (metric == "last"):
        paths = [p for p in Path(checkpoint_dir).rglob(f"*{metric}*.ckpt")]
        assert len(paths) > 0

        return paths[0]
    for ckpt_path in Path(checkpoint_dir).rglob(f"*{metric}*.ckpt"):
        match = pattern.search(ckpt_path.name)

        if match:
            val_loss = float(match.group(1))

            if mode == "min":
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_path
            else:
                if val_loss > best_val_loss:
                    best_val_loss = val_loss
                    best_ckpt = ckpt_path
    return best_ckpt


def load_best_model(
    path_training: Path, strict: bool = True, metrics=None, option="mm_sp"
) -> tuple[
    SupervisedCD | MMSupervisedSP | MMSupervisedForecast, DictConfig, Path | None
]:
    config_path = Path(path_training).joinpath(".hydra").joinpath("config.yaml")

    config = OmegaConf.load(config_path)

    # Load checkpoint and clean compiled model keys
    ckpt_path = get_best_checkpoint(path_training, metric="best")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = remove_compiled_prefix(checkpoint["state_dict"])

    model = instantiate(config.model)
    if metrics is None:
        metrics = MetricCollection(
            {
                name: instantiate(metric_cfg)
                for name, metric_cfg in config.metrics.items()
            }
        )
    train_config = instantiate(config.train.train_config, metrics=metrics)
    train_loss = instantiate(config.train.train_config.loss)
    train_config.loss = train_loss

    if option == "mm_sp":
        pl_module: SupervisedCD | SupervisedRepr | MMSupervisedSP = instantiate(
            config.lightning_module,
            model=model,
            train_config=train_config,
            compile=False,
        )
        pl_module.load_state_dict(checkpoint["state_dict"], strict=strict)
    elif option == "forecast":
        pl_module: SupervisedCD | SupervisedRepr | MMSupervisedSP = instantiate(
            config.lightning_module,
            model=model,
            train_config=train_config,
            shallow_forecaster=instantiate(config.lightning_module.shallow_forecaster),
            compile=False,
        )
        pl_module.load_state_dict(checkpoint["state_dict"], strict=strict)
    return pl_module, config, ckpt_path


def run_test(datamodule, pl_module: SupervisedCD, device="cpu"):
    pl_module = pl_module.to(device)
    trainer = Trainer(accelerator=device, devices=1, detect_anomaly=False)
    trainer.test(pl_module, datamodule=datamodule)
    # for batch_idx, batch in enumerate(test_dataloader):
    #     batch = batch.to_device(device)
    #     with torch.no_grad():
    #         pl_module.test_step(batch, batch_idx=batch_idx)
    return pl_module.test_metrics.compute()


def run_mm_test(
    datamodule: MMSPDataModule | MMForecastDataModule,
    test_dataloader: DataLoader,
    pl_module: MMSupervisedSP | MMSupervisedForecast,
    device="cpu",
):
    pl_module = pl_module.to(device)
    print(f"module on {device}")
    assert pl_module.test_metrics is not None
    trainer = Trainer(accelerator=device, devices=1, detect_anomaly=False)
    trainer.test(pl_module, datamodule=datamodule)
    return pl_module.test_metrics.compute()


def assess_one_model(
    path_training: Path,
    dataset_path: str,
    path_dir_csv: str,
):
    pl_module, config, ckpt_path = load_best_model(path_training=path_training)
    pl_module.freeze()
    pl_module.eval()
    datamodule = instantiate(
        config.datamodule,
        dataset_path=dataset_path,
        path_dir_csv=path_dir_csv,
        batch_size=1,
    )
    datamodule.setup("test")

    return run_test(datamodule.test_dataloader(), pl_module)


def extract_path_config(path):
    print(path)
    path_config = path.parents[3].joinpath(".hydra").joinpath("config.yaml")
    assert path_config.exists()
    return path_config


def extract_hp(config_path):
    d_params = {}
    print(config_path)
    config = OmegaConf.load(config_path)
    d_params["d_model"] = config.shared.d_model
    d_params["batch_size"] = config.train.train_config.batch_size
    d_params["lr"] = config.train.train_config.lr
    d_params["model"] = config.model._target_
    try:
        d_params["attention"] = config.model.temporal_encoder.layer.attn_block._target_
        d_params["is_causal"] = config.model.is_causal
        d_params["num_layers"] = config.model.temporal_encoder.num_layers
    except (AttributeError, KeyError):
        d_params["attention"] = None
    # d_params["model"]=instantiate(config.model)
    try:
        d_params["s2_max_len"] = config.datamodule.max_len_s2
    except (AttributeError, KeyError):
        d_params["s2_max"] = config.datamodule.max_len
    try:
        d_params["expe"] = config.datamodule.fold_expe
    except (AttributeError, KeyError):
        d_params["expe"] = None
    d_params["loss"] = type(instantiate(config.train.train_config.loss)).__name__
    d_params["decoder"] = type(instantiate(config.model.last_layer)).__name__
    # d_params["training_path"]=config_path
    return pd.Series(d_params, name=config_path.parents[1])
