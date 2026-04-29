import multiprocessing
import os
import signal
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from cd_mm_sits.callback.monitor import TrackBestValLoss
from cd_mm_sits.data.datamodule.brads1_datamodule import BradS1DataModule
from cd_mm_sits.data.datamodule.mm_zarr_sp_datamodule import MMSPDataModule
from cd_mm_sits.data.datamodule.solar_panels_datamodule import SPDataModule
from cd_mm_sits.lightning_module.mm_supervised_sp_detection import MMSupervisedSP
from cd_mm_sits.lightning_module.supervised_cd import SupervisedCD
from cd_mm_sits.lightning_module.supervised_sits_representations import SupervisedRepr


@hydra.main(
    config_path="../config/", config_name="supervised_cd.yaml", version_base="1.2"
)
def main(config: DictConfig):
    seed_everything(config.seed, workers=True)
    model = instantiate(config.model)
    if config.metrics is not None:
        metrics = MetricCollection(
            {
                name: hydra.utils.instantiate(metric_cfg)
                for name, metric_cfg in config.metrics.items()
            }
        )
    else:
        metrics = None
    hydra_cfg = HydraConfig.get()
    train_config = instantiate(config.train.train_config, metrics=metrics)

    callbacks = [instantiate(cb_conf) for _, cb_conf in config.callback.items()]
    logger = [
        instantiate(logg_conf, save_dir=hydra_cfg.runtime.output_dir)
        for _, logg_conf in config.logger.items()
    ]
    pl_module: SupervisedCD | SupervisedRepr | MMSupervisedSP = instantiate(
        config.lightning_module,
        model=model,
        train_config=train_config,
    )
    datamodule: SPDataModule | BradS1DataModule | MMSPDataModule = instantiate(
        config.datamodule
    )
    datamodule.setup("fit")
    best_loss_callback = TrackBestValLoss(config.optimized_metric)
    if config.train.slurm_restart:
        print("Automatic restart")
        strategy = DDPStrategy(
            cluster_environment=SLURMEnvironment(
                requeue_signal=signal.SIGHUP, auto_requeue=True
            ),
            find_unused_parameters=True,
        )
    else:
        strategy = config.train.trainer.strategy

    trainer: Trainer = instantiate(
        config.train.trainer,
        logger=logger,
        callbacks=[best_loss_callback] + callbacks,
        strategy=strategy,
    )
    trainer.fit(
        pl_module,
        datamodule,
        ckpt_path=config.ckpt_path,
    )

    print("Best val_loss:", best_loss_callback.best_val_loss)
    if config.run_test:
        trainer.test(pl_module, datamodule=datamodule, ckpt_path="best")
        metrics = pl_module.save_test_metrics
        torch.save(metrics, Path(os.getcwd()).joinpath("test_metrics.pt"))
    return best_loss_callback.best_val_loss


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    multiprocessing.set_start_method("spawn")
    main()
