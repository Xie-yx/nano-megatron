from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from model.GPT_model import GPT, GPTConfig


@dataclass
class TrainConfig:
	data_dir: str
	out_dir: str = "out"
	csv_log_name: str = "loss_log.csv"
	batch_size: int = 8
	max_steps: int = 200
	eval_interval: int = 50
	eval_iters: int = 20
	log_interval: int = 10
	save_interval: int = 100
	learning_rate: float = 3e-4
	weight_decay: float = 0.1
	grad_clip: float = 1.0
	seed: int = 42
	device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> tuple[TrainConfig, GPTConfig]:
	parser = argparse.ArgumentParser(description="Minimal single-GPU GPT trainer")
	parser.add_argument("--data-dir", default="data/shakespeare", help="包含 train.bin 和 val.bin 的目录")
	parser.add_argument("--out-dir", default="out", help="checkpoint 输出目录")
	parser.add_argument("--csv-log-name", default="loss_log.csv", help="训练日志 CSV 文件名，保存在 out-dir 下")
	parser.add_argument("--batch-size", type=int, default=8, help="每步 batch size")
	parser.add_argument("--max-steps", type=int, default=200, help="训练步数")
	parser.add_argument("--eval-interval", type=int, default=50, help="评估间隔")
	parser.add_argument("--eval-iters", type=int, default=20, help="每次评估的 batch 数")
	parser.add_argument("--log-interval", type=int, default=10, help="日志间隔")
	parser.add_argument("--save-interval", type=int, default=100, help="checkpoint 间隔")
	parser.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW 学习率")
	parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW 权重衰减")
	parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
	parser.add_argument("--seed", type=int, default=42, help="随机种子")
	parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="训练设备")
	parser.add_argument("--vocab-size", type=int, default=50257, help="GPT-2 BPE 词表大小")

	parser.add_argument("--num-layers", type=int, default=4, help="Transformer 层数")
	parser.add_argument("--hidden-size", type=int, default=256, help="隐藏维度")
	parser.add_argument("--ffn-hidden-size", type=int, default=1024, help="MLP 中间层维度")
	parser.add_argument("--num-attention-heads", type=int, default=4, help="注意力头数")
	parser.add_argument("--max-seq-len", type=int, default=128, help="训练时 block size")
	parser.add_argument("--embedding-dropout", type=float, default=0.1, help="embedding dropout")
	parser.add_argument("--attention-dropout", type=float, default=0.1, help="attention dropout")
	parser.add_argument("--residual-dropout", type=float, default=0.1, help="residual dropout")
	parser.add_argument("--use-bias", action="store_true", help="线性层使用 bias")

	args = parser.parse_args()

	train_config = TrainConfig(
		data_dir=args.data_dir,
		out_dir=args.out_dir,
		csv_log_name=args.csv_log_name,
		batch_size=args.batch_size,
		max_steps=args.max_steps,
		eval_interval=args.eval_interval,
		eval_iters=args.eval_iters,
		log_interval=args.log_interval,
		save_interval=args.save_interval,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		grad_clip=args.grad_clip,
		seed=args.seed,
		device=args.device,
	)
	model_config = GPTConfig(
		num_layers=args.num_layers,
		hidden_size=args.hidden_size,
		ffn_hidden_size=args.ffn_hidden_size,
		num_attention_heads=args.num_attention_heads,
		vocab_size=args.vocab_size,
		max_seq_len=args.max_seq_len,
		embedding_dropout=args.embedding_dropout,
		attention_dropout=args.attention_dropout,
		residual_dropout=args.residual_dropout,
		use_bias=args.use_bias,
	)
	return train_config, model_config


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def load_token_files(data_dir: str) -> tuple[np.memmap, np.memmap]:
	data_path = Path(data_dir)
	train_path = data_path / "train.bin"
	val_path = data_path / "val.bin"
	if not train_path.exists() or not val_path.exists():
		raise FileNotFoundError(
			f"未找到预处理后的数据文件，请先运行 {data_path / 'prepare.py'} 生成 train.bin 和 val.bin。"
		)
	train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
	val_tokens = np.memmap(val_path, dtype=np.uint16, mode="r")
	if train_tokens.size == 0 or val_tokens.size == 0:
		raise ValueError("train.bin 或 val.bin 为空，无法训练。")
	return train_tokens, val_tokens


def init_csv_logger(csv_path: Path) -> None:
	csv_path.parent.mkdir(parents=True, exist_ok=True)
	if csv_path.exists():
		return
	with csv_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(
			file,
			fieldnames=[
				"event",
				"step",
				"train_loss",
				"eval_train_loss",
				"eval_val_loss",
				"learning_rate",
				"step_time_ms",
			],
		)
		writer.writeheader()


def append_csv_log(
	csv_path: Path,
	event: str,
	step: int,
	learning_rate: float,
	train_loss: float | None = None,
	eval_train_loss: float | None = None,
	eval_val_loss: float | None = None,
	step_time_ms: float | None = None,
) -> None:
	with csv_path.open("a", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(
			file,
			fieldnames=[
				"event",
				"step",
				"train_loss",
				"eval_train_loss",
				"eval_val_loss",
				"learning_rate",
				"step_time_ms",
			],
		)
		writer.writerow(
			{
				"event": event,
				"step": step,
				"train_loss": "" if train_loss is None else f"{train_loss:.8f}",
				"eval_train_loss": "" if eval_train_loss is None else f"{eval_train_loss:.8f}",
				"eval_val_loss": "" if eval_val_loss is None else f"{eval_val_loss:.8f}",
				"learning_rate": f"{learning_rate:.8e}",
				"step_time_ms": "" if step_time_ms is None else f"{step_time_ms:.4f}",
			}
		)


def get_batch(token_ids: np.memmap, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
	if token_ids.shape[0] <= block_size + 1:
		raise ValueError("token 序列长度必须大于 max_seq_len。")
	starts = torch.randint(0, token_ids.shape[0] - block_size - 1, (batch_size,))
	input_ids = torch.stack([
		torch.from_numpy(np.asarray(token_ids[start:start + block_size], dtype=np.int64))
		for start in starts.tolist()
	])
	labels = torch.stack([
		torch.from_numpy(np.asarray(token_ids[start + 1:start + block_size + 1], dtype=np.int64))
		for start in starts.tolist()
	])
	return input_ids.to(device), labels.to(device)


@torch.no_grad()
def estimate_loss(
	model: GPT,
	train_tokens: np.memmap,
	val_tokens: np.memmap,
	train_config: TrainConfig,
	model_config: GPTConfig,
) -> dict[str, float]:
	model.eval()
	losses: dict[str, float] = {}
	for split_name, split_tokens_tensor in (("train", train_tokens), ("val", val_tokens)):
		split_losses = torch.zeros(train_config.eval_iters)
		for index in range(train_config.eval_iters):
			input_ids, labels = get_batch(
				split_tokens_tensor,
				train_config.batch_size,
				model_config.max_seq_len,
				train_config.device,
			)
			logits = model(input_ids)
			split_losses[index] = F.cross_entropy(
				logits.view(-1, logits.size(-1)),
				labels.view(-1),
			).item()
		losses[split_name] = split_losses.mean().item()
	model.train()
	return losses


def save_checkpoint(
	model: GPT,
	optimizer: torch.optim.Optimizer,
	train_config: TrainConfig,
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
			"tokenizer_name": "gpt2",
		},
		checkpoint_path,
	)
	return checkpoint_path


def main() -> None:
	train_config, model_config = parse_args()
	set_seed(train_config.seed)
	out_dir = Path(train_config.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	csv_path = out_dir / train_config.csv_log_name
	init_csv_logger(csv_path)

	train_tokens, val_tokens = load_token_files(train_config.data_dir)
	if train_tokens.shape[0] < model_config.max_seq_len + 2 or val_tokens.shape[0] < model_config.max_seq_len + 2:
		raise ValueError("train.bin 或 val.bin 太短，至少需要大于 max_seq_len + 1 个 token。")

	model = GPT(model_config).to(train_config.device)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=train_config.learning_rate,
		weight_decay=train_config.weight_decay,
	)

	print(f"device: {train_config.device}")
	print("tokenizer: gpt2")
	print(f"vocab size: {model_config.vocab_size}")
	print(f"train tokens: {train_tokens.shape[0]}, val tokens: {val_tokens.shape[0]}")
	print(f"csv log: {csv_path}")

	model.train()
	for step in range(1, train_config.max_steps + 1):
		start_time = time.time()
		input_ids, labels = get_batch(
			train_tokens,
			train_config.batch_size,
			model_config.max_seq_len,
			train_config.device,
		)
		logits = model(input_ids)
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		if train_config.grad_clip > 0:
			torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
		optimizer.step()
		learning_rate = optimizer.param_groups[0]["lr"]
		elapsed_ms = (time.time() - start_time) * 1000

		if step % train_config.log_interval == 0 or step == 1:
			print(f"step {step:06d} | train loss {loss.item():.4f} | step time {elapsed_ms:.2f} ms")
			append_csv_log(
				csv_path,
				event="train",
				step=step,
				train_loss=loss.item(),
				learning_rate=learning_rate,
				step_time_ms=elapsed_ms,
			)

		if train_config.eval_interval > 0 and step % train_config.eval_interval == 0:
			losses = estimate_loss(model, train_tokens, val_tokens, train_config, model_config)
			print(
				f"step {step:06d} | eval train loss {losses['train']:.4f} | eval val loss {losses['val']:.4f}"
			)
			append_csv_log(
				csv_path,
				event="eval",
				step=step,
				eval_train_loss=losses["train"],
				eval_val_loss=losses["val"],
				learning_rate=learning_rate,
			)

		if train_config.save_interval > 0 and step % train_config.save_interval == 0:
			checkpoint_path = save_checkpoint(model, optimizer, train_config, model_config, step)
			print(f"saved checkpoint to {checkpoint_path}")

	final_checkpoint = save_checkpoint(model, optimizer, train_config, model_config, train_config.max_steps)
	print(f"training finished, final checkpoint saved to {final_checkpoint}")


if __name__ == "__main__":
	main()





