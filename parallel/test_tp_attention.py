import torch
import torch.nn as nn

from parallel.global_vars import get_args
from parallel.initialize import initialize_megatron
from parallel import mpu
from parallel.model.GPT_model import ParallelSelfAttention


TEST_SEED = 3031
ATOL = 1e-5
RTOL = 1e-5


def _device() -> torch.device:
    return torch.device(get_args().device)


def _rank() -> int:
    return torch.distributed.get_rank()


def _world_size() -> int:
    return torch.distributed.get_world_size()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch, max diff={diff}")


class ReferenceSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attn_dropout: float, resid_dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(resid_dropout)
        self.norm_factor = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.norm_factor
        scores = scores - attention_mask * 10000.0
        probs = self.softmax(scores)
        probs = self.attention_dropout(probs)

        output = probs @ v
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.residual_dropout(self.out_proj(output))
        return output


def _build_reference_attention(args) -> ReferenceSelfAttention:
    torch.manual_seed(TEST_SEED)
    module = ReferenceSelfAttention(
        hidden_size=args.hidden_size,
        num_heads=args.num_attention_heads,
        attn_dropout=args.attention_dropout,
        resid_dropout=args.residual_dropout,
    )
    return module.to(_device())


def _build_input(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(TEST_SEED + 1)
    return torch.randn(batch_size, seq_len, hidden_size, device=_device())


def _build_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _copy_attention_params(ref_attn: ReferenceSelfAttention, tp_attn: ParallelSelfAttention) -> None:
    tp_rank = mpu.get_tp_rank()

    shard_size = tp_attn.q_proj.output_size_per_partition
    shard_start = tp_rank * shard_size
    shard_end = shard_start + shard_size

    with torch.no_grad():
        tp_attn.q_proj.weight.copy_(ref_attn.q_proj.weight[shard_start:shard_end].to(tp_attn.q_proj.weight.device))
        tp_attn.q_proj.bias.copy_(ref_attn.q_proj.bias[shard_start:shard_end].to(tp_attn.q_proj.bias.device))

        tp_attn.k_proj.weight.copy_(ref_attn.k_proj.weight[shard_start:shard_end].to(tp_attn.k_proj.weight.device))
        tp_attn.k_proj.bias.copy_(ref_attn.k_proj.bias[shard_start:shard_end].to(tp_attn.k_proj.bias.device))

        tp_attn.v_proj.weight.copy_(ref_attn.v_proj.weight[shard_start:shard_end].to(tp_attn.v_proj.weight.device))
        tp_attn.v_proj.bias.copy_(ref_attn.v_proj.bias[shard_start:shard_end].to(tp_attn.v_proj.bias.device))

        out_shard = tp_attn.out_proj.input_size_per_partition
        out_start = tp_rank * out_shard
        out_end = out_start + out_shard
        tp_attn.out_proj.weight.copy_(ref_attn.out_proj.weight[:, out_start:out_end].to(tp_attn.out_proj.weight.device))
        tp_attn.out_proj.bias.copy_(ref_attn.out_proj.bias.to(tp_attn.out_proj.bias.device))


def test_parallel_attention_forward_backward() -> None:
    args = get_args()
    batch_size = 2
    seq_len = 4

    ref_attn = _build_reference_attention(args)
    tp_attn = ParallelSelfAttention(
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=nn.init.xavier_normal_,
        layer_number=1,
    ).to(_device())
    _copy_attention_params(ref_attn, tp_attn)

    ref_attn.eval()
    tp_attn.eval()

    x_ref = _build_input(batch_size, seq_len, args.hidden_size).clone().detach().requires_grad_(True)
    x_tp = x_ref.detach().clone().requires_grad_(True)
    attention_mask = _build_causal_mask(batch_size, seq_len, _device())

    y_ref = ref_attn(x_ref, attention_mask)
    y_tp = tp_attn(x_tp, attention_mask)
    _assert_close("attention forward", y_tp, y_ref)

    loss_ref = (y_ref ** 2).sum()
    loss_tp = (y_tp ** 2).sum()
    loss_ref.backward()
    loss_tp.backward()

    tp_rank = mpu.get_tp_rank()
    shard_size = tp_attn.q_proj.output_size_per_partition
    shard_start = tp_rank * shard_size
    shard_end = shard_start + shard_size

    _assert_close("q_proj weight grad", tp_attn.q_proj.weight.grad, ref_attn.q_proj.weight.grad[shard_start:shard_end])
    _assert_close("q_proj bias grad", tp_attn.q_proj.bias.grad, ref_attn.q_proj.bias.grad[shard_start:shard_end])
    _assert_close("k_proj weight grad", tp_attn.k_proj.weight.grad, ref_attn.k_proj.weight.grad[shard_start:shard_end])
    _assert_close("k_proj bias grad", tp_attn.k_proj.bias.grad, ref_attn.k_proj.bias.grad[shard_start:shard_end])
    _assert_close("v_proj weight grad", tp_attn.v_proj.weight.grad, ref_attn.v_proj.weight.grad[shard_start:shard_end])
    _assert_close("v_proj bias grad", tp_attn.v_proj.bias.grad, ref_attn.v_proj.bias.grad[shard_start:shard_end])

    out_shard = tp_attn.out_proj.input_size_per_partition
    out_start = tp_rank * out_shard
    out_end = out_start + out_shard
    _assert_close("out_proj weight grad", tp_attn.out_proj.weight.grad, ref_attn.out_proj.weight.grad[:, out_start:out_end])
    _assert_close("out_proj bias grad", tp_attn.out_proj.bias.grad, ref_attn.out_proj.bias.grad)
    _assert_close("attention input grad", x_tp.grad, x_ref.grad)

    if _rank() == 0:
        print("ParallelSelfAttention forward/backward passed.", flush=True)


def main() -> None:
    initialize_megatron()
    args = get_args()

    if _world_size() != args.tensor_model_parallel_size:
        raise ValueError(
            "For this minimal test, world_size must equal tensor_model_parallel_size: "
            f"world_size={_world_size()}, tp={args.tensor_model_parallel_size}"
        )

    test_parallel_attention_forward_backward()
    torch.distributed.barrier()

    if _rank() == 0:
        print("All TP attention tests passed.", flush=True)


if __name__ == "__main__":
    main()
