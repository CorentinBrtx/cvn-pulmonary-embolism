import logging
from logging.handlers import WatchedFileHandler
import sys
from typing import Dict


def setup_logger(log_file: str):
    _logger = logging.getLogger("train")
    _logger.setLevel(logging.INFO)
    if len(_logger.handlers) == 0:
        formatter = logging.Formatter("%(asctime)s | %(message)s")
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        _logger.addHandler(stream_handler)
        file_handler = WatchedFileHandler(log_file)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
    return _logger


class SummaryWriter:
    def __init__(self, nb_epochs: int, nb_batchs_train: int, nb_batchs_val: int, logger) -> None:
        self.nb_epochs = nb_epochs
        self.nb_batchs_train = nb_batchs_train
        self.nb_batchs_val = nb_batchs_val
        self.epoch = 0
        self.logger = logger

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, mode: str, i_batch: int, metrics: Dict[str, float]) -> None:
        if mode.lower() == "train":
            nb_batchs = self.nb_batchs_train
        else:
            nb_batchs = self.nb_batchs_val
        summary = (
            f"{mode.title()} Epoch {self.epoch}/{self.nb_epochs} | Batch {i_batch}/{nb_batchs} | "
        )
        for metric_name, metric_value in metrics.items():
            summary += f"{metric_name.title()} {metric_value:.4f} | "
        self.logger.info(summary[:-2])
