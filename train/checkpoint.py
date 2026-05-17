from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.tokenizer import GPT2_TOKENIZER_SPEC
from model.GPT_model import GPT, GPTConfig


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_config: Any,
    model_config: GPTConfig,
    step: int,
) -> Path:
    out_dir = Path(train_config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"checkpoint_step_{step:06d}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "train_config": asdict(train_config),
            "tokenizer_name": GPT2_TOKENIZER_SPEC.name,
            "token_dtype": np.dtype(GPT2_TOKENIZER_SPEC.token_dtype).name,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[GPT, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = GPTConfig(**checkpoint["model_config"])
    model = GPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, checkpoint
