import torch
import torch.nn.functional as F

from parallel.global_vars import get_args
from parallel.initialize import initialize_megatron
from parallel import mpu
from parallel.model.GPT_model import ParallelGPT


TEST_SEED = 4040


def _device() -> torch.device:
    return torch.device(get_args().device)


def _rank() -> int:
    return torch.distributed.get_rank()


def _world_size() -> int:
    return torch.distributed.get_world_size()


def _build_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    torch.manual_seed(TEST_SEED)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, position_ids, labels


def _build_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains inf or nan values")


def test_parallel_gpt_forward_shapes() -> None:
    args = get_args()
    batch_size = 2
    seq_len = min(8, args.max_position_embeddings)
    device = _device()

    model = ParallelGPT(parallel_output=False).to(device)
    model.eval()

    input_ids, position_ids, _ = _build_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )
    attention_mask = _build_causal_mask(batch_size, seq_len, device)

    logits = model(input_ids, position_ids, attention_mask)
    expected_shape = (batch_size, seq_len, args.padded_vocab_size)
    if logits.shape != expected_shape:
        raise AssertionError(f"full logits shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}")
    _assert_finite("full logits", logits)

    logits_parallel = model(
        input_ids,
        position_ids,
        attention_mask,
        forward_method_parallel_output=True,
    )
    expected_parallel_shape = (
        batch_size,
        seq_len,
        args.padded_vocab_size // args.tensor_model_parallel_size,
    )
    if logits_parallel.shape != expected_parallel_shape:
        raise AssertionError(
            f"parallel logits shape mismatch: got {tuple(logits_parallel.shape)}, expected {expected_parallel_shape}"
        )
    _assert_finite("parallel logits", logits_parallel)

    if _rank() == 0:
        print("ParallelGPT forward shape test passed.", flush=True)


def test_parallel_gpt_backward() -> None:
    args = get_args()
    batch_size = 2
    seq_len = min(8, args.max_position_embeddings)
    device = _device()

    torch.manual_seed(TEST_SEED + 1)
    model = ParallelGPT(parallel_output=False).to(device)
    model.train()

    input_ids, position_ids, labels = _build_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=args.vocab_size,
        device=device,
    )
    attention_mask = _build_causal_mask(batch_size, seq_len, device)

    logits = model(input_ids, position_ids, attention_mask)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    _assert_finite("gpt loss", loss.detach())
    loss.backward()

    embedding_grad = model.language_model.embedding.word_embeddings.weight.grad
    if embedding_grad is None:
        raise AssertionError("word embedding gradient is missing")
    _assert_finite("word embedding grad", embedding_grad)

    first_block = model.language_model.transformer.blocks[0]
    if first_block.attn.q_proj.weight.grad is None:
        raise AssertionError("attention q_proj gradient is missing")
    if first_block.mlp.dense_4h_to_h.weight.grad is None:
        raise AssertionError("mlp output projection gradient is missing")

    _assert_finite("attention q_proj grad", first_block.attn.q_proj.weight.grad)
    _assert_finite("mlp output grad", first_block.mlp.dense_4h_to_h.weight.grad)

    if _rank() == 0:
        print("ParallelGPT backward smoke test passed.", flush=True)


def main() -> None:
    initialize_megatron(
        args_defaults={
            "num_layers": 2,
            "hidden_size": 256,
            "ffn_hidden_size": 1024,
            "num_attention_heads": 4,
            "max_seq_len": 16,
        }
    )
    args = get_args()

    if _world_size() != args.tensor_model_parallel_size:
        raise ValueError(
            "For this minimal test, world_size must equal tensor_model_parallel_size: "
            f"world_size={_world_size()}, tp={args.tensor_model_parallel_size}"
        )

    test_parallel_gpt_forward_shapes()
    torch.distributed.barrier()
    test_parallel_gpt_backward()
    torch.distributed.barrier()

    if _rank() == 0:
        print("All ParallelGPT smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()