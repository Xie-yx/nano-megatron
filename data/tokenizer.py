from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tiktoken


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    vocab_size: int
    token_dtype: type[np.generic]


GPT2_TOKENIZER_SPEC = TokenizerSpec(
    name="gpt2",
    vocab_size=50257,
    token_dtype=np.uint16,
)


class GPT2Tokenizer:
    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(GPT2_TOKENIZER_SPEC.name)
        self.spec = GPT2_TOKENIZER_SPEC

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def token_dtype(self) -> type[np.generic]:
        return self.spec.token_dtype

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(text)

    def decode(self, token_ids: Iterable[int]) -> str:
        return self._encoding.decode(list(token_ids))


def get_gpt2_tokenizer() -> GPT2Tokenizer:
    return GPT2Tokenizer()
