from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.tokenizer import get_gpt2_tokenizer
from train.checkpoint import load_checkpoint


@torch.no_grad()
def generate_text(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str | None = None,
) -> str:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = get_gpt2_tokenizer()
    model, _ = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    input_ids = tokenizer.encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.max_seq_len :]
        logits = model(x_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

    return tokenizer.decode(x[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_step_xxxxxx.pt")
    parser.add_argument("--prompt", default="To be, or not to be", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; <=0 means greedy")
    parser.add_argument("--top-k", type=int, default=None, help="Optional top-k sampling")
    parser.add_argument("--device", default=None, help="cpu or cuda; default auto")
    args = parser.parse_args()

    text = generate_text(
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
    )
    print(text)


if __name__ == "__main__":
    main()
