import json
import time
from typing import Dict, Any

from src.ui.callbacks import TrainingCallback

class MetricsLoggerCallback(TrainingCallback):
    """Listens to the training loop and saves a JSON artifact at the end of the run."""
    
    def __init__(self, model_name: str, param_count: int):
        self.model_name = model_name
        self.param_count = param_count
        self.start_time = 0.0
        self.metrics: Dict[str, Any] = {
            "model_name": self.model_name,
            "parameter_count": self.param_count,
            "total_training_time_s": 0.0,
            "test_accuracy": 0.0,
            "epoch_history": []
        }

    def on_train_begin(self, total_epochs: int, total_batches: int):
        self.start_time = time.perf_counter()

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        # Expects metrics like: {"train_loss": 0.5, "val_loss": 0.6, "val_acc": 80.0}
        row = {"epoch": epoch}
        row.update(metrics)
        self.metrics["epoch_history"].append(row)

    def on_train_end(self, test_accuracy: float = 0.0):
        self.metrics["total_training_time_s"] = time.perf_counter() - self.start_time
        self.metrics["test_accuracy"] = test_accuracy
        
        # HYDRA MAGIC: Because Hydra set the CWD to outputs/YYYY-MM-DD/HH-MM-SS/,
        # dumping to a local filename automatically saves it in the run's timestamped folder!
        with open("metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=4)