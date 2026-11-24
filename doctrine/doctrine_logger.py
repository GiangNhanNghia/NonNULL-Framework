"""
NonNULL Framework V1.0 - Telemetry & Logging Adapter
File: doctrine/doctrine_logger.py

Provides colorized console logging, file logging, and HarmonyLogger adapter
for TensorBoard / Weights & Biases telemetry tracking.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


class GenesisFormatter(logging.Formatter):
    """Custom console log formatter for EULER telemetry output."""

    FORMATS = (
        {
            logging.DEBUG: Fore.CYAN + "[DEBUG] " + Style.RESET_ALL + "%(message)s",
            logging.INFO: Fore.GREEN + "[INFO]  " + Style.RESET_ALL + "%(message)s",
            logging.WARNING: Fore.YELLOW + "[WARN]  " + Style.RESET_ALL + "%(message)s",
            logging.ERROR: Fore.RED + "[ERROR] " + Style.RESET_ALL + "%(message)s",
            logging.CRITICAL: Fore.RED + Style.BRIGHT + "[FATAL] " + Style.RESET_ALL + "%(message)s",
        }
        if HAS_COLOR
        else {}
    )

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, "[%(levelname)s] %(message)s")
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(name: str = "EULER", log_file: str = "genesis.log") -> logging.Logger:
    """Initializes and returns a non-duplicating logger instance."""
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(GenesisFormatter() if HAS_COLOR else logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(ch)

        try:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
            log.addHandler(fh)
        except OSError:
            pass

    return log


logger = setup_logger()


def log_genesis(message: str) -> None:
    """Highlights major system lifecycle transitions in the console."""
    if HAS_COLOR:
        print(Fore.MAGENTA + Style.BRIGHT + "\n=== [GENESIS EVENT] ===")
        print(Fore.MAGENTA + f"[*] {message}")
        print(Fore.MAGENTA + "=======================\n")
    else:
        logger.info(f"=== GENESIS: {message} ===")


class HarmonyLogger:
    """
    Telemetry adapter (Mô-đun Dương 5 - Tam Tiêu) interfacing with local JSONL logs,
    TensorBoard, and Weights & Biases (W&B) if installed.
    """

    def __init__(
        self,
        log_dir: str = "./output/logs",
        wandb_project: Optional[str] = None,
        use_tensorboard: bool = False,
    ) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, "telemetry_metrics.jsonl")
        self.tb_writer = None
        self.wandb_active = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=self.log_dir)
            except ImportError:
                pass

        if wandb_project:
            try:
                import wandb
                if wandb.run is None:
                    wandb.init(project=wandb_project, dir=self.log_dir)
                self.wandb_active = True
            except ImportError:
                pass

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Records step-level loss components, gate activation g_t, and phase metrics."""
        record = {"step": step, **metrics}
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, step)

        if self.wandb_active:
            import wandb
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.tb_writer is not None:
            self.tb_writer.close()