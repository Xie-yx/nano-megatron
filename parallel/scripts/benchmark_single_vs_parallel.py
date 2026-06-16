from __future__ import annotations

import argparse
import csv
import os
import shlex
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_TRAIN_ENTRY = REPO_ROOT / "_single_gpu_baseline" / "sctipts" / "train_single_gpu_benchmark.py"
PARALLEL_ENTRY = "parallel.pretrain_GPT_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-GPU vs single-node multi-GPU training (pair or sweep mode)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("pair", "sweep"),
        default="sweep",
        help="pair: run one parallel config; sweep: run a TP/DP matrix.",
    )
    parser.add_argument("--single-gpu", type=str, default="0", help="GPU id for single-GPU run.")
    parser.add_argument(
        "--parallel-gpus",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated GPU ids for parallel run (e.g. '0,1').",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=8,
        help="Number of processes for torchrun in parallel mode.",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=4,
        help="Tensor model parallel size for parallel mode.",
    )
    parser.add_argument(
        "--data-dir-single",
        type=str,
        default="_single_gpu_baseline/data/shakespeare",
        help="Data dir used by single-GPU trainer.",
    )
    parser.add_argument(
        "--data-dir-parallel",
        type=str,
        default="parallel/data/shakespeare",
        help="Data dir used by parallel trainer.",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="Training steps for each run.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Ignore first N logged steps when averaging step time.")
    parser.add_argument("--log-interval", type=int, default=1, help="Logging interval for both runs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Single-GPU batch size.")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="Parallel per-rank micro batch size.")
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--ffn-hidden-size", type=int, default=4096)
    parser.add_argument("--num-attention-heads", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sweep-gpu-counts",
        type=str,
        default="2,4,8",
        help="Comma-separated nproc_per_node values for sweep mode.",
    )
    parser.add_argument(
        "--sweep-tp-candidates",
        type=str,
        default="1,2,4,8",
        help="Comma-separated tensor parallel candidates for sweep mode.",
    )
    parser.add_argument(
        "--target-global-batch",
        type=int,
        default=None,
        help="If set, sweep mode auto-adjusts micro-batch per case to match this global batch as closely as possible.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run folder name under benchmark/runs/. Defaults to timestamp.",
    )
    return parser.parse_args()


def _run_command(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    print("\n==> Running:")
    print(" ".join(shlex.quote(token) for token in cmd))
    print(f"    log: {log_path}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def _load_train_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV log not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("event") == "train":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No train rows found in {csv_path}")
    return rows


def _to_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value is None or value == "":
        return None
    return float(value)


def summarize_csv(csv_path: Path, warmup_steps: int) -> dict[str, float]:
    rows = _load_train_rows(csv_path)

    step_times = [_to_float(row, "step_time_ms") for row in rows]
    step_times = [v for v in step_times if v is not None]
    if not step_times:
        raise RuntimeError(f"No step_time_ms values found in {csv_path}")

    trimmed = step_times[warmup_steps:] if len(step_times) > warmup_steps else step_times
    avg_step_time_ms = statistics.mean(trimmed)

    max_allocated_mb = max((_to_float(row, "max_allocated_mb") or 0.0) for row in rows)
    max_reserved_mb = max((_to_float(row, "max_reserved_mb") or 0.0) for row in rows)

    return {
        "logged_steps": float(len(step_times)),
        "avg_step_time_ms": avg_step_time_ms,
        "max_allocated_mb": max_allocated_mb,
        "max_reserved_mb": max_reserved_mb,
    }


def _common_model_args(args: argparse.Namespace, include_hidden_dropout: bool) -> list[str]:
    common = [
        "--max-steps",
        str(args.max_steps),
        "--log-interval",
        str(args.log_interval),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--seed",
        str(args.seed),
        "--num-layers",
        str(args.num_layers),
        "--hidden-size",
        str(args.hidden_size),
        "--ffn-hidden-size",
        str(args.ffn_hidden_size),
        "--num-attention-heads",
        str(args.num_attention_heads),
        "--max-seq-len",
        str(args.max_seq_len),
        "--embedding-dropout",
        "0.0",
        "--attention-dropout",
        "0.0",
        "--residual-dropout",
        "0.0",
    ]
    if include_hidden_dropout:
        common.extend(["--hidden-dropout", "0.0"])
    return common


def _parse_csv_ints(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_gpu_ids(raw: str) -> list[str]:
    values = [token.strip() for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("parallel-gpus must provide at least one GPU id.")
    return values


def _tokens_per_sec(global_batch: int, seq_len: int, avg_step_time_ms: float) -> float:
    if avg_step_time_ms <= 0:
        return 0.0
    return (global_batch * seq_len) / (avg_step_time_ms / 1000.0)


def _parallel_case_name(nproc: int, tp: int, dp: int) -> str:
    return f"n{nproc}_tp{tp}_dp{dp}"


def _run_single(args: argparse.Namespace, run_root: Path, single_common_args: list[str]) -> dict[str, Any]:
    single_out = run_root / "single"
    single_csv = single_out / "loss_log.csv"

    single_cmd = [
        sys.executable,
        str(SINGLE_TRAIN_ENTRY),
        "--data-dir",
        args.data_dir_single,
        "--out-dir",
        str(single_out),
        "--csv-log-name",
        "loss_log.csv",
        "--batch-size",
        str(args.batch_size),
    ] + single_common_args

    single_env = os.environ.copy()
    single_env["CUDA_VISIBLE_DEVICES"] = args.single_gpu

    _run_command(single_cmd, single_env, run_root / "single.log")
    single_stats = summarize_csv(single_csv, args.warmup_steps)

    return {
        "mode": "single",
        "case": "single",
        "nproc": 1,
        "tp": 1,
        "dp": 1,
        "micro_batch": args.batch_size,
        "global_batch": args.batch_size,
        "avg_step_time_ms": single_stats["avg_step_time_ms"],
        "peak_allocated_mb": single_stats["max_allocated_mb"],
        "peak_reserved_mb": single_stats["max_reserved_mb"],
        "logged_steps": int(single_stats["logged_steps"]),
        "tokens_per_sec": _tokens_per_sec(args.batch_size, args.max_seq_len, single_stats["avg_step_time_ms"]),
    }


def _build_sweep_cases(args: argparse.Namespace) -> list[dict[str, int]]:
    available_gpus = _parse_gpu_ids(args.parallel_gpus)
    gpu_counts = sorted(set(_parse_csv_ints(args.sweep_gpu_counts)))
    tp_candidates = sorted(set(_parse_csv_ints(args.sweep_tp_candidates)))

    target_global_batch = args.target_global_batch
    if target_global_batch is None:
        target_global_batch = args.batch_size

    cases: list[dict[str, int]] = []
    for nproc in gpu_counts:
        if nproc < 2 or nproc > len(available_gpus):
            continue
        for tp in tp_candidates:
            if tp < 1 or tp > nproc:
                continue
            if nproc % tp != 0:
                continue

            dp = nproc // tp
            micro_batch = max(1, target_global_batch // dp)
            global_batch = micro_batch * dp
            cases.append(
                {
                    "nproc": nproc,
                    "tp": tp,
                    "dp": dp,
                    "micro_batch": micro_batch,
                    "global_batch": global_batch,
                }
            )

    if not cases:
        raise RuntimeError("No valid sweep cases generated. Check --parallel-gpus, --sweep-gpu-counts, and --sweep-tp-candidates.")
    return cases


def _run_parallel_case(
    args: argparse.Namespace,
    run_root: Path,
    parallel_common_args: list[str],
    nproc: int,
    tp: int,
    micro_batch: int,
) -> dict[str, Any]:
    dp = nproc // tp
    case = _parallel_case_name(nproc, tp, dp)
    case_out = run_root / f"parallel_{case}"
    case_csv = case_out / "loss_log.csv"

    available_gpu_ids = _parse_gpu_ids(args.parallel_gpus)
    selected_gpu_ids = available_gpu_ids[:nproc]

    parallel_cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc),
        "-m",
        PARALLEL_ENTRY,
        "--tensor-model-parallel-size",
        str(tp),
        "--data-dir",
        args.data_dir_parallel,
        "--out-dir",
        str(case_out),
        "--csv-log-name",
        "loss_log.csv",
        "--micro-batch-size",
        str(micro_batch),
    ] + parallel_common_args

    parallel_env = os.environ.copy()
    parallel_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_gpu_ids)

    _run_command(parallel_cmd, parallel_env, run_root / f"parallel_{case}.log")
    stats = summarize_csv(case_csv, args.warmup_steps)

    global_batch = micro_batch * dp
    return {
        "mode": "parallel",
        "case": case,
        "nproc": nproc,
        "tp": tp,
        "dp": dp,
        "micro_batch": micro_batch,
        "global_batch": global_batch,
        "avg_step_time_ms": stats["avg_step_time_ms"],
        "peak_allocated_mb": stats["max_allocated_mb"],
        "peak_reserved_mb": stats["max_reserved_mb"],
        "logged_steps": int(stats["logged_steps"]),
        "tokens_per_sec": _tokens_per_sec(global_batch, args.max_seq_len, stats["avg_step_time_ms"]),
    }


def _write_summary(rows: list[dict[str, Any]], summary_path: Path) -> None:
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mode",
                "case",
                "nproc",
                "tp",
                "dp",
                "micro_batch",
                "global_batch",
                "avg_step_time_ms",
                "peak_allocated_mb",
                "peak_reserved_mb",
                "tokens_per_sec",
                "speedup_vs_single",
                "peak_alloc_ratio_vs_single",
                "logged_steps",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _print_summary(rows: list[dict[str, Any]], run_root: Path, summary_path: Path) -> None:
    print("\n===== Benchmark Summary =====")
    print(f"run_dir: {run_root}")
    print(
        "case              nproc tp dp  gbs  avg_step_ms   peak_alloc_mb    tok/s   speedup"
    )
    for row in rows:
        print(
            f"{row['case']:<16}"
            f" {row['nproc']:>5}"
            f" {row['tp']:>2}"
            f" {row['dp']:>2}"
            f" {row['global_batch']:>4}"
            f" {row['avg_step_time_ms']:>12.3f}"
            f" {row['peak_allocated_mb']:>15.3f}"
            f" {row['tokens_per_sec']:>8.1f}"
            f" {row['speedup_vs_single']:>8.3f}"
        )
    print(f"summary_csv: {summary_path}")


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = REPO_ROOT / "benchmark" / "runs" / run_name

    single_common_args = _common_model_args(args, include_hidden_dropout=False)
    parallel_common_args = _common_model_args(args, include_hidden_dropout=True)

    all_rows: list[dict[str, Any]] = []
    single_row = _run_single(args, run_root, single_common_args)
    all_rows.append(single_row)

    if args.mode == "pair":
        pair_row = _run_parallel_case(
            args,
            run_root,
            parallel_common_args,
            nproc=args.nproc_per_node,
            tp=args.tensor_model_parallel_size,
            micro_batch=args.micro_batch_size,
        )
        all_rows.append(pair_row)
    else:
        sweep_cases = _build_sweep_cases(args)
        print("\nPlanned sweep cases:")
        for case in sweep_cases:
            print(
                f"- {_parallel_case_name(case['nproc'], case['tp'], case['dp'])} "
                f"(micro_batch={case['micro_batch']}, global_batch={case['global_batch']})"
            )

        for case in sweep_cases:
            row = _run_parallel_case(
                args,
                run_root,
                parallel_common_args,
                nproc=case["nproc"],
                tp=case["tp"],
                micro_batch=case["micro_batch"],
            )
            all_rows.append(row)

    single_step_ms = single_row["avg_step_time_ms"]
    single_peak_alloc_mb = max(single_row["peak_allocated_mb"], 1e-9)

    for row in all_rows:
        row["speedup_vs_single"] = single_step_ms / row["avg_step_time_ms"]
        row["peak_alloc_ratio_vs_single"] = row["peak_allocated_mb"] / single_peak_alloc_mb

    summary_path = run_root / "summary.csv"
    _write_summary(all_rows, summary_path)
    _print_summary(all_rows, run_root, summary_path)


if __name__ == "__main__":
    main()
