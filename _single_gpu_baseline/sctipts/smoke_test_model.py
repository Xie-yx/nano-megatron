import os
import sys
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.GPT_model import GPT, GPTConfig



def main():
    config = GPTConfig(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=512,
        num_attention_heads=4,
        vocab_size=100,
        max_seq_len=32,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        residual_dropout=0.1,
        use_bias=True,
        layernorm_epsilon=1e-5,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    model = GPT(config).to(device)
    model.train()

    batch_size = 4
    seq_len = 16

    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    labels = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    logits = model(input_ids)

    print(f"input_ids.shape = {input_ids.shape}")
    print(f"logits.shape    = {logits.shape}")

    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"logits shape mismatch: got {logits.shape}, expected {expected_shape}"
    )

    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    print(f"loss = {loss.item():.6f}")

    model.zero_grad(set_to_none=True)
    loss.backward()

    no_grad_params = []
    total_grad_norm = 0.0

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                no_grad_params.append(name)
            else:
                total_grad_norm += param.grad.norm().item()

    assert len(no_grad_params) == 0, f"Some params have no grad: {no_grad_params}"

    print(f"total_grad_norm = {total_grad_norm:.6f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()

    print("Smoke test passed.")


if __name__ == "__main__":
    main()