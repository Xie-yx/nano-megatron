import torch
import torch.nn as nn

from parallel.global_vars import get_args
from parallel.initialize import initialize_megatron
from parallel import mpu
from parallel.model.GPT_model import ParallelMLP


TEST_SEED = 2025
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


class ReferenceMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _build_reference_mlp(hidden_size: int, ffn_hidden_size: int) -> ReferenceMLP:
    torch.manual_seed(TEST_SEED)
    return ReferenceMLP(hidden_size, ffn_hidden_size).to(_device())


def _build_input(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(TEST_SEED + 1)
    return torch.randn(batch_size, seq_len, hidden_size, device=_device())


def _copy_mlp_params(ref_mlp: ReferenceMLP, tp_mlp: ParallelMLP) -> None:
    tp_rank = mpu.get_tp_rank()

    fc1_shard = tp_mlp.dense_h_to_4h.output_size_per_partition
    fc1_start = tp_rank * fc1_shard
    fc1_end = fc1_start + fc1_shard

    fc2_shard = tp_mlp.dense_4h_to_h.input_size_per_partition
    fc2_start = tp_rank * fc2_shard
    fc2_end = fc2_start + fc2_shard

    with torch.no_grad():
        tp_mlp.dense_h_to_4h.weight.copy_(
            ref_mlp.fc1.weight[fc1_start:fc1_end].to(tp_mlp.dense_h_to_4h.weight.device)
        )
        tp_mlp.dense_h_to_4h.bias.copy_(
            ref_mlp.fc1.bias[fc1_start:fc1_end].to(tp_mlp.dense_h_to_4h.bias.device)
        )
        tp_mlp.dense_4h_to_h.weight.copy_(
            ref_mlp.fc2.weight[:, fc2_start:fc2_end].to(tp_mlp.dense_4h_to_h.weight.device)
        )
        tp_mlp.dense_4h_to_h.bias.copy_(
            ref_mlp.fc2.bias.to(tp_mlp.dense_4h_to_h.bias.device)
        )


def test_parallel_mlp_forward_backward() -> None:
    args = get_args()
    batch_size = 2
    seq_len = 4

    ref_mlp = _build_reference_mlp(args.hidden_size, args.ffn_hidden_size)
    tp_mlp = ParallelMLP(
        init_method=nn.init.xavier_normal_,
        output_layer_init_method=nn.init.xavier_normal_,
    ).to(_device())
    _copy_mlp_params(ref_mlp, tp_mlp)

    x_ref = _build_input(batch_size, seq_len, args.hidden_size).clone().detach().requires_grad_(True)
    x_tp = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref_mlp(x_ref)
    y_tp = tp_mlp(x_tp)
    _assert_close("mlp forward", y_tp, y_ref)

    loss_ref = (y_ref ** 2).sum()
    loss_tp = (y_tp ** 2).sum()
    loss_ref.backward()
    loss_tp.backward()

    tp_rank = mpu.get_tp_rank()
    fc1_shard = tp_mlp.dense_h_to_4h.output_size_per_partition
    fc1_start = tp_rank * fc1_shard
    fc1_end = fc1_start + fc1_shard

    fc2_shard = tp_mlp.dense_4h_to_h.input_size_per_partition
    fc2_start = tp_rank * fc2_shard
    fc2_end = fc2_start + fc2_shard

    _assert_close(
        "mlp fc1 weight grad",
        tp_mlp.dense_h_to_4h.weight.grad,
        ref_mlp.fc1.weight.grad[fc1_start:fc1_end],
    )
    _assert_close(
        "mlp fc1 bias grad",
        tp_mlp.dense_h_to_4h.bias.grad,
        ref_mlp.fc1.bias.grad[fc1_start:fc1_end],
    )
    _assert_close(
        "mlp fc2 weight grad",
        tp_mlp.dense_4h_to_h.weight.grad,
        ref_mlp.fc2.weight.grad[:, fc2_start:fc2_end],
    )
    _assert_close(
        "mlp fc2 bias grad",
        tp_mlp.dense_4h_to_h.bias.grad,
        ref_mlp.fc2.bias.grad,
    )
    _assert_close("mlp input grad", x_tp.grad, x_ref.grad)

    if _rank() == 0:
        print("ParallelMLP forward/backward passed.", flush=True)


def main() -> None:
    initialize_megatron()
    args = get_args()

    if _world_size() != args.tensor_model_parallel_size:
        raise ValueError(
            "For this minimal test, world_size must equal tensor_model_parallel_size: "
            f"world_size={_world_size()}, tp={args.tensor_model_parallel_size}"
        )

    test_parallel_mlp_forward_backward()
    torch.distributed.barrier()

    if _rank() == 0:
        print("All TP MLP tests passed.", flush=True)


if __name__ == "__main__":
    main()
