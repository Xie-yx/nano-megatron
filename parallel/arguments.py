from __future__ import annotations

import argparse
import os

import torch


DEFAULT_VOCAB_SIZE = 50257


def parse_args(
	extra_args_provider=None,
	defaults: dict[str, object] | None = None,
	ignore_unknown_args: bool = False,
	argv: list[str] | None = None,
):
	"""Parse runtime arguments for nano-megatron.

	This is a deliberately small subset of Megatron's argument system:
	one parser, a few argument groups, post-processing for derived fields,
	and validation for TP/DP-related constraints.
	"""
	parser = argparse.ArgumentParser(
		description="nano-megatron arguments",
		allow_abbrev=False,
	)

	parser = _add_distributed_args(parser)
	parser = _add_model_args(parser)
	parser = _add_training_args(parser)
	parser = _add_data_args(parser)
	parser = _add_runtime_args(parser)

	if extra_args_provider is not None:
		parser = extra_args_provider(parser)

	if defaults:
		parser.set_defaults(**defaults)

	if ignore_unknown_args:
		args, _ = parser.parse_known_args(argv)
	else:
		args = parser.parse_args(argv)

	_postprocess_args(args)
	_validate_args(args)
	return args


def _add_distributed_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	group = parser.add_argument_group("distributed")
	group.add_argument(
		"--distributed-backend",
		type=str,
		default="nccl",
		help="torch.distributed backend",
	)
	group.add_argument(
		"--dist-timeout",
		type=int,
		default=30,
		help="torch.distributed process group initialization timeout in minutes",
	)
	group.add_argument(
		"--tensor-model-parallel-size",
		type=int,
		default=1,
		help="Tensor parallel world size",
	)
	group.add_argument(
		"--rank",
		type=int,
		default=None,
		help="Global rank. Defaults to env RANK or 0.",
	)
	group.add_argument(
		"--world-size",
		type=int,
		default=None,
		help="Global world size. Defaults to env WORLD_SIZE or 1.",
	)
	group.add_argument(
		"--local-rank",
		type=int,
		default=None,
		help="Local rank on current node. Defaults to env LOCAL_RANK or rank modulo device count.",
	)
	return parser


def _add_model_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	group = parser.add_argument_group("model")
	group.add_argument("--num-layers", type=int, default=4, help="Number of transformer blocks")
	group.add_argument("--hidden-size", type=int, default=256, help="Transformer hidden size")
	group.add_argument("--ffn-hidden-size", type=int, default=1024, help="MLP intermediate size")
	group.add_argument("--num-attention-heads", type=int, default=4, help="Number of attention heads")
	group.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, help="Vocabulary size")
	group.add_argument("--max-seq-len", type=int, default=128, help="Maximum sequence length")
	group.add_argument("--embedding-dropout", type=float, default=0.1, help="Embedding dropout")
	group.add_argument("--attention-dropout", type=float, default=0.1, help="Attention dropout")
	group.add_argument("--residual-dropout", type=float, default=0.1, help="Residual dropout")
	group.add_argument("--layernorm-epsilon", type=float, default=1e-5, help="LayerNorm epsilon")
	group.add_argument("--use-bias", action="store_true", help="Enable bias in linear layers")
	return parser


def _add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	group = parser.add_argument_group("training")
	group.add_argument("--micro-batch-size", type=int, default=8, help="Per-rank micro batch size")
	group.add_argument(
		"--global-batch-size",
		type=int,
		default=None,
		help="Global batch size. Defaults to micro-batch-size * data-parallel-size.",
	)
	group.add_argument("--max-steps", type=int, default=200, help="Number of training steps")
	group.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate")
	group.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay")
	group.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold")
	group.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser


def _add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	group = parser.add_argument_group("data")
	group.add_argument("--data-dir", type=str, default="data/shakespeare", help="Directory containing train.bin and val.bin")
	return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	group = parser.add_argument_group("runtime")
	group.add_argument("--out-dir", type=str, default="out", help="Checkpoint and log output directory")
	group.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
	group.add_argument("--log-interval", type=int, default=10, help="Logging interval in steps")
	group.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval in steps")
	group.add_argument("--save-interval", type=int, default=100, help="Checkpoint interval in steps")
	group.add_argument("--eval-iters", type=int, default=20, help="Number of batches per evaluation")
	group.add_argument("--csv-log-name", type=str, default="loss_log.csv", help="CSV filename for training logs")
	group.add_argument(
		"--device",
		type=str,
		default=None,
		help="Runtime device. Defaults to cuda:<local_rank> when CUDA is available, else cpu.",
	)
	return parser


def _postprocess_args(args) -> None:
	args.rank = _resolve_rank(args.rank)
	args.world_size = _resolve_world_size(args.world_size)
	args.tensor_model_parallel_size = min(args.tensor_model_parallel_size, args.world_size)

	device_count = torch.cuda.device_count()
	if args.local_rank is None:
		env_local_rank = os.getenv("LOCAL_RANK")
		if env_local_rank is not None:
			args.local_rank = int(env_local_rank)
		elif device_count > 0:
			args.local_rank = args.rank % device_count
		else:
			args.local_rank = 0

	args.data_parallel_size = args.world_size // args.tensor_model_parallel_size
	args.pipeline_model_parallel_size = 1
	args.model_parallel_size = args.tensor_model_parallel_size

	if args.global_batch_size is None:
		args.global_batch_size = args.micro_batch_size * args.data_parallel_size

	if args.device is None:
		if torch.cuda.is_available():
			args.device = f"cuda:{args.local_rank}"
		else:
			args.device = "cpu"


def _validate_args(args) -> None:
	if args.rank < 0:
		raise ValueError(f"rank must be non-negative, got {args.rank}.")
	if args.world_size < 1:
		raise ValueError(f"world_size must be at least 1, got {args.world_size}.")
	if args.tensor_model_parallel_size < 1:
		raise ValueError(
			"tensor_model_parallel_size must be at least 1, "
			f"got {args.tensor_model_parallel_size}."
		)
	if args.world_size % args.tensor_model_parallel_size != 0:
		raise ValueError(
			"world_size must be divisible by tensor_model_parallel_size: "
			f"world_size={args.world_size}, "
			f"tensor_model_parallel_size={args.tensor_model_parallel_size}."
		)
	if args.micro_batch_size < 1:
		raise ValueError(f"micro_batch_size must be at least 1, got {args.micro_batch_size}.")
	if args.global_batch_size < 1:
		raise ValueError(f"global_batch_size must be at least 1, got {args.global_batch_size}.")
	if args.hidden_size % args.num_attention_heads != 0:
		raise ValueError(
			"hidden_size must be divisible by num_attention_heads: "
			f"hidden_size={args.hidden_size}, num_attention_heads={args.num_attention_heads}."
		)
	if args.num_attention_heads % args.tensor_model_parallel_size != 0:
		raise ValueError(
			"num_attention_heads must be divisible by tensor_model_parallel_size: "
			f"num_attention_heads={args.num_attention_heads}, "
			f"tensor_model_parallel_size={args.tensor_model_parallel_size}."
		)


def _resolve_rank(rank: int | None) -> int:
	if rank is not None:
		return rank
	return int(os.getenv("RANK", "0"))


def _resolve_world_size(world_size: int | None) -> int:
	if world_size is not None:
		return world_size
	return int(os.getenv("WORLD_SIZE", "1"))




