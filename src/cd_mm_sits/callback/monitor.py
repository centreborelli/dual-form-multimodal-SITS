from lightning import Callback


class TrackBestValLoss(Callback):
    def __init__(self, monitor="val_loss"):
        self.monitor = monitor
        self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        current = trainer.callback_metrics.get(self.monitor)
        if current is not None:
            current = current.item()
            if current < self.best_val_loss:
                self.best_val_loss = current
