import numpy as np
import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """A token stream dataset that returns contiguous next-token samples."""

    def __init__(self, tokens, seq_length):
        if len(tokens) <= seq_length:
            raise ValueError(
                f"Token stream is too short: got {len(tokens)} tokens for seq_length={seq_length}."
            )
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return len(self.tokens) - self.seq_length

    def __getitem__(self, idx):
        sample = np.asarray(
            self.tokens[idx : idx + self.seq_length + 1],
            dtype=np.int64,
        )
        return torch.from_numpy(sample)
