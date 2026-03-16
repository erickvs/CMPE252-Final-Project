class TrainingCallback:
    """
    Abstract base class for tracking the training lifecycle.
    Keeps the PyTorch training loop strictly decoupled from any specific UI or logging logic.
    """
    def on_train_begin(self, total_epochs: int, total_batches: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_batch_end(self, batch_idx: int, loss: float):
        pass

    def on_epoch_end(self, epoch: int, metrics: dict):
        pass

    def on_train_end(self, test_accuracy: float = 0.0):
        pass