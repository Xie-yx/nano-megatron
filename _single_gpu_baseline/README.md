# nano-megatron

Author: Xieyx

A lightweight toy implementation inspired by Megatron-LM.  
Current status: single-GPU training with GPT-2 tokenizer is working, including:

- data preparation
- training with train/val loss logging
- checkpoint saving
- checkpoint loading
- text generation from checkpoint

## Current Validation Flow

The current single-GPU baseline uses:

- dataset: `tinyshakespeare`
- tokenizer: `tiktoken` GPT-2 BPE
- vocab size: `50257`
- token storage dtype: `uint16`

The recommended validation order is:

1. prepare data
2. run single-GPU training
3. verify checkpoints are written
4. run generation from a checkpoint
5. verify checkpoint loading / resume behavior

---

## 1. Environment

Recommended Python version:

- `Python 3.10`

### Conda

```bash
conda create -n nano-megatron python=3.10 -y
conda activate nano-megatron
```

### Install PyTorch

Check your GPU driver first:

```bash
nvidia-smi
```

Example for CUDA 12.1:

```bash
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -f https://mirrors.nju.edu.cn/pytorch/whl/cu128
```

Example for CUDA 11.8:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

### Install project dependencies

```bash
pip install -r requirements.txt
```

### Verify environment

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

---

## 2. Prepare Data

Generate `train.bin` and `val.bin` from Tiny Shakespeare:

```bash
python data/shakespeare/prepare.py
```

Expected artifacts:

- `data/shakespeare/input.txt`
- `data/shakespeare/train.bin`
- `data/shakespeare/val.bin`

---

## 3. Single-GPU Training

### Minimal smoke training

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/shakespeare_smoke \
  --batch-size 4 \
  --max-steps 20 \
  --eval-interval 10 \
  --save-interval 10 \
  --max-seq-len 64 \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 512 \
  --num-attention-heads 4
```

### Recommended single-GPU baseline run

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/shakespeare_run \
  --csv-log-name loss_log.csv \
  --batch-size 8 \
  --max-steps 500 \
  --eval-interval 50 \
  --save-interval 100 \
  --max-seq-len 128 \
  --num-layers 4 \
  --hidden-size 256 \
  --ffn-hidden-size 1024 \
  --num-attention-heads 4
```

Training outputs:

- checkpoints under `out/shakespeare_run/`
- csv log under `out/shakespeare_run/loss_log.csv`

Useful checks:

```bash
ls -lh out/shakespeare_run
tail -n 20 out/shakespeare_run/loss_log.csv
```

---

## 4. Inference / Generation

Generate text from a saved checkpoint:

```bash
python inference/generate.py \
  --checkpoint out/shakespeare_run/checkpoint_step_000100.pt \
  --prompt "To be, or not to be" \
  --max-new-tokens 100 \
  --temperature 1.0 \
  --top-k 20 \
  --device cuda
```

For greedy decoding:

```bash
python inference/generate.py \
  --checkpoint out/shakespeare_run/checkpoint_step_000100.pt \
  --prompt "To be, or not to be" \
  --max-new-tokens 80 \
  --temperature 0 \
  --device cuda
```

---

## 5. Checkpoint Verification

### Verify that checkpoints can be loaded

The generation command above already verifies:

- checkpoint file can be read
- model config can be restored
- model weights can be loaded
- tokenizer metadata is consistent enough for decode

### Manual Python check

You can also manually verify checkpoint content:

```bash
python - <<'PY'
import torch
ckpt = torch.load("out/shakespeare_run/checkpoint_step_000100.pt", map_location="cpu")
print("step:", ckpt["step"])
print("tokenizer_name:", ckpt.get("tokenizer_name"))
print("token_dtype:", ckpt.get("token_dtype"))
print("model_config keys:", sorted(ckpt["model_config"].keys()))
print("train_config keys:", sorted(ckpt["train_config"].keys()))
PY
```

---

## 6. Resume Verification

Current code supports resuming training from a saved checkpoint through `--resume`.

Recommended validation procedure:

1. run an initial short training job
2. confirm checkpoint exists
3. resume from an intermediate checkpoint
4. verify new checkpoints and logs continue from the saved step

### Step 1: initial run

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/resume_test \
  --batch-size 8 \
  --max-steps 120 \
  --eval-interval 40 \
  --save-interval 60 \
  --max-seq-len 128 \
  --num-layers 4 \
  --hidden-size 256 \
  --ffn-hidden-size 1024 \
  --num-attention-heads 4
```

This should create:

- `out/resume_test/checkpoint_step_000060.pt`
- `out/resume_test/checkpoint_step_000120.pt`

### Step 2: resume from step 60 and continue to step 180

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/resume_test \
  --resume out/resume_test/checkpoint_step_000060.pt \
  --batch-size 8 \
  --max-steps 180 \
  --eval-interval 40 \
  --save-interval 60 \
  --max-seq-len 128 \
  --num-layers 4 \
  --hidden-size 256 \
  --ffn-hidden-size 1024 \
  --num-attention-heads 4
```

Expected behavior:

- script prints `resuming from: ...`
- script prints `resume step: 60`
- training continues from step `61`
- new checkpoint `checkpoint_step_000180.pt` is produced

### Step 3: verify the resumed checkpoint can generate

```bash
python inference/generate.py \
  --checkpoint out/resume_test/checkpoint_step_000180.pt \
  --prompt "The king" \
  --max-new-tokens 60 \
  --temperature 1.0 \
  --top-k 20 \
  --device cuda
```

### Step 4: inspect checkpoint metadata

```bash
python - <<'PY'
import torch
ckpt = torch.load("out/resume_test/checkpoint_step_000180.pt", map_location="cpu")
print("resume step:", ckpt["step"])
print("tokenizer_name:", ckpt.get("tokenizer_name"))
print("token_dtype:", ckpt.get("token_dtype"))
PY
```

---

## 7. What Counts as Success for Single-GPU Baseline

At this stage, the single-GPU baseline is considered validated if:

- training runs without crashing
- `train loss` decreases
- `eval val loss` is reported
- checkpoints are saved
- checkpoints can be loaded for generation
- generated text runs end-to-end from the saved checkpoint

---

## 8. Next Recommended Step

After the single-GPU flow is fully validated on Linux server, the next step is:

- cleanly finish single-GPU checkpoint/resume flow
- then begin TP primitives:
  - tensor-parallel mappings
  - `ColumnParallelLinear`
  - `RowParallelLinear`
  - TP correctness tests
