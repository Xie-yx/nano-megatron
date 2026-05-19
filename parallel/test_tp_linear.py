import torch
import torch.nn as nn

from parallel.global_vars import get_args
from parallel.initialize import initialize_megatron
from parallel import mpu
from parallel.mpu.layers import ColumnParallelLinear, RowParallelLinear


TEST_SEED = 1234
ATOL = 1e-5
RTOL = 1e-5


def _device() -> torch.device:
    args = get_args()
    return torch.device(args.device)


def _rank() -> int:
    return torch.distributed.get_rank()


def _world_size() -> int:
    return torch.distributed.get_world_size()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch, max diff={diff}")


def _build_reference_linear(input_size: int, output_size: int, bias: bool) -> nn.Linear:
    torch.manual_seed(TEST_SEED)
    layer = nn.Linear(input_size, output_size, bias=bias)
    return layer.to(_device())


def _build_input(batch_size: int, seq_len: int, hidden_size: int) -> torch.Tensor:
    torch.manual_seed(TEST_SEED + 1)
    x = torch.randn(batch_size, seq_len, hidden_size, device=_device())
    return x


def _copy_column_parallel_params(
    ref_layer: nn.Linear,
    tp_layer: ColumnParallelLinear,
) -> None:
    rank = mpu.get_tp_rank()
    shard_size = tp_layer.output_size_per_partition
    start = rank * shard_size
    end = start + shard_size

    with torch.no_grad():
        tp_layer.weight.copy_(ref_layer.weight[start:end].to(tp_layer.weight.device))
        if tp_layer.bias is not None:
            tp_layer.bias.copy_(ref_layer.bias[start:end].to(tp_layer.bias.device))


def _copy_row_parallel_params(
    ref_layer: nn.Linear,
    tp_layer: RowParallelLinear,
) -> None:
    rank = mpu.get_tp_rank()
    shard_size = tp_layer.input_size_per_partition
    start = rank * shard_size
    end = start + shard_size

    with torch.no_grad():
        tp_layer.weight.copy_(ref_layer.weight[:, start:end].to(tp_layer.weight.device))
        if tp_layer.bias is not None:
            tp_layer.bias.copy_(ref_layer.bias.to(tp_layer.bias.device))


def test_column_parallel_forward_backward() -> None:
    batch_size = 2
    seq_len = 4
    input_size = 8
    output_size = 12

    ref_layer = _build_reference_linear(input_size, output_size, bias=True)
    tp_layer = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        gather_output=True,
    ).to(_device())
    _copy_column_parallel_params(ref_layer, tp_layer)

    x_ref = _build_input(batch_size, seq_len, input_size).clone().detach().requires_grad_(True)
    x_tp = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref_layer(x_ref)
    y_tp = tp_layer(x_tp)
    _assert_close("column forward", y_tp, y_ref)

    loss_ref = (y_ref ** 2).sum()
    loss_tp = (y_tp ** 2).sum()
    loss_ref.backward()
    loss_tp.backward()

    shard_size = tp_layer.output_size_per_partition
    start = mpu.get_tp_rank() * shard_size
    end = start + shard_size
    _assert_close("column weight grad", tp_layer.weight.grad, ref_layer.weight.grad[start:end])
    _assert_close("column bias grad", tp_layer.bias.grad, ref_layer.bias.grad[start:end])
    _assert_close("column input grad", x_tp.grad, x_ref.grad)

    if _rank() == 0:
        print("ColumnParallelLinear forward/backward passed.", flush=True)


def test_row_parallel_forward_backward() -> None:
    batch_size = 2
    seq_len = 4
    input_size = 8
    output_size = 12

    ref_layer = _build_reference_linear(input_size, output_size, bias=True)
    tp_layer = RowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        input_is_parallel=False,
    ).to(_device())
    _copy_row_parallel_params(ref_layer, tp_layer)

    x_ref = _build_input(batch_size, seq_len, input_size).clone().detach().requires_grad_(True)
    x_tp = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref_layer(x_ref)
    y_tp = tp_layer(x_tp)
    _assert_close("row forward", y_tp, y_ref)

    loss_ref = (y_ref ** 2).sum()
    loss_tp = (y_tp ** 2).sum()
    loss_ref.backward()
    loss_tp.backward()

    shard_size = tp_layer.input_size_per_partition
    start = mpu.get_tp_rank() * shard_size
    end = start + shard_size
    _assert_close("row weight grad", tp_layer.weight.grad, ref_layer.weight.grad[:, start:end])
    _assert_close("row bias grad", tp_layer.bias.grad, ref_layer.bias.grad)
    _assert_close("row input grad", x_tp.grad, x_ref.grad)

    if _rank() == 0:
        print("RowParallelLinear forward/backward passed.", flush=True)


def main() -> None:
    initialize_megatron()
    args = get_args()

    if _world_size() != args.tensor_model_parallel_size:
        raise ValueError(
            "For this minimal test, world_size must equal tensor_model_parallel_size: "
            f"world_size={_world_size()}, tp={args.tensor_model_parallel_size}"
        )

    if _rank() == 0:
        print(
            f"Running TP linear tests with world_size={_world_size()}, "
            f"tp_rank={mpu.get_tp_rank()}, device={args.device}",
            flush=True,
        )

    test_column_parallel_forward_backward()
    torch.distributed.barrier()
    test_row_parallel_forward_backward()
    torch.distributed.barrier()

    if _rank() == 0:
        print("All TP linear tests passed.", flush=True)


if __name__ == "__main__":
    main()
