import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

@dataclass
class TrainConfig:
    micro_batch_size: int = 4
    global_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    warmup_steps: int = 1000
    eval_interval: int = 1000
    save_interval: int = 1000
    log_interval: int = 100
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"