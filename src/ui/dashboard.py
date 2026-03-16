import psutil
from collections import deque
import torch

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich.console import Console

from src.ui.callbacks import TrainingCallback

class RichDashboardCallback(TrainingCallback):
    """
    A live-updating, non-blocking Terminal User Interface built with 'rich'.
    Uses Apple Silicon specific APIs to track MPS memory allocation.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.loss_history = deque(maxlen=40)
        self.epoch_metrics = []
        
        # 1. Setup Progress Bars
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        
        # 2. Build the Layout Grid
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=1)
        )
        self.layout["left_panel"].split_column(
            Layout(name="progress"),
            Layout(name="sparkline", size=4)
        )
        
        # 3. Start Live Display (screen=True uses alternate terminal buffer)
        # This keeps the professor's terminal history clean when training finishes.
        self.live = Live(self.layout, refresh_per_second=8, screen=True)

    def _generate_sparkline(self) -> str:
        """Generates a text-based mini chart for recent losses."""
        if not self.loss_history:
            return "Awaiting data..."
        chars = " ▂▃▄▅▆▇█"
        min_val, max_val = min(self.loss_history), max(self.loss_history)
        range_val = max_val - min_val if max_val > min_val else 1
        
        sparkline = "".join(chars[int(((val - min_val) / range_val) * 7)] for val in self.loss_history)
        return f"[bold magenta]{sparkline}[/] (Latest: {self.loss_history[-1]:.4f})"

    def _generate_metrics_table(self) -> Table:
        table = Table(expand=True, box=None)
        table.add_column("Epoch", justify="center", style="cyan")
        table.add_column("LR", justify="right", style="blue")
        table.add_column("Train Loss", justify="right", style="magenta")
        table.add_column("Val Loss", justify="right", style="green")
        table.add_column("Val Acc", justify="right", style="yellow")
        
        # Show only last 8 epochs to prevent overflow
        for row in self.epoch_metrics[-8:]:
            # Safely get LR in case older logs don't have it
            lr_str = f"{row.get('lr', 0):.6f}" if 'lr' in row else "N/A"
            table.add_row(
                str(row['epoch']), 
                lr_str,
                f"{row['train_loss']:.4f}", 
                f"{row['val_loss']:.4f}", 
                f"{row['val_acc']:.2f}%"
            )
        return table

    def _update_layout(self):
        """Pushes current state to the UI grid."""
        # Header
        header = f"[bold white]CIFAR-10 Comparative Analysis[/] | Model: [bold cyan]{self.model_name}[/]"
        self.layout["header"].update(Panel(Align.center(header), style="on dark_blue"))

        # Main Panels
        self.layout["progress"].update(Panel(self.progress, title="[bold]Training Progress", border_style="blue"))
        
        spark_text = Text.from_markup(f"Rolling Batch Loss (Last 40):\n{self._generate_sparkline()}")
        self.layout["sparkline"].update(Panel(Align.center(spark_text), border_style="magenta"))

        self.layout["right_panel"].update(Panel(self._generate_metrics_table(), title="[bold]Validation History", border_style="green"))

        # Footer (Hardware Telemetry)
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent()
        
        # Flex: Native Apple Silicon memory tracking!
        mps_mem = "N/A"
        if torch.backends.mps.is_available():
            try:
                # current_allocated_memory returns bytes
                mps_mem = f"{torch.mps.current_allocated_memory() / (1024**2):.1f} MB"
            except Exception:
                mps_mem = "Tracking Error"

        footer_text = f"⚙️ CPU: {cpu}% | 🧠 RAM: {ram}% | 🍏 MPS VRAM: {mps_mem}"
        self.layout["footer"].update(Panel(Align.center(footer_text), border_style="yellow"))

    # --- Interface Methods (Triggered by the Engine) ---
    def on_train_begin(self, total_epochs: int, total_batches: int):
        self.epoch_task = self.progress.add_task("Global Epochs", total=total_epochs)
        self.batch_task = self.progress.add_task("Current Epoch", total=total_batches)
        self.live.start()
        self._update_layout()

    def on_epoch_begin(self, epoch: int):
        self.progress.reset(self.batch_task)
        self._update_layout()

    def on_batch_end(self, batch_idx: int, loss: float):
        self.loss_history.append(loss)
        self.progress.advance(self.batch_task)
        if batch_idx % 5 == 0: # Throttle UI updates slightly to save CPU
            self._update_layout()

    def on_epoch_end(self, epoch: int, metrics: dict):
        self.progress.advance(self.epoch_task)
        metrics["epoch"] = epoch
        self.epoch_metrics.append(metrics)
        self._update_layout()

    def on_train_end(self, test_accuracy: float = 0.0):
        self.live.stop()
        # Print a final summary to standard out so it remains after UI closes
        Console().print(f"\n[bold green]✅ {self.model_name} Training Complete! Test Accuracy: {test_accuracy:.2f}%[/]")
        Console().print(self._generate_metrics_table())