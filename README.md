# nano-megatron

`nano-megatron` is a small educational GPT training project inspired by nanoGPT, Megatron-LM, and CodeGeeX. The project keeps a validated single-GPU baseline under `_single_gpu_baseline/`, and develops tensor/data parallel training under `parallel/`.

The current parallel implementation supports:

- Tensor parallel process groups and communication primitives
- `ColumnParallelLinear` and `RowParallelLinear`
- Tensor-parallel MLP and self-attention
- `VocabParallelEmbedding`
- Vocab-parallel cross entropy
- `ParallelGPT` training with TP, plus a minimal TP+DP path through PyTorch DDP

Multi-node training is not enabled by default in the examples because the test servers currently restrict non-SSH TCP ports. The code is structured for `torchrun` multi-node launch once the cluster network allows the rendezvous port and NCCL communication.

## Directory Layout

```text
nano-megatron/
  _single_gpu_baseline/     # validated single-GPU reference implementation
  parallel/
    arguments.py            # Megatron-style runtime arguments
    initialize.py           # distributed and model-parallel initialization
    training.py             # simplified CodeGeeX-style training skeleton
    pretrain_GPT.py         # GPT pretraining task entry
    data/                   # GPT-2-tokenized Shakespeare data pipeline
    model/                  # ParallelGPT implementation
    mpu/                    # tensor-parallel state, mappings, layers, loss
    tests/                  # TP correctness tests
  requirements.txt
```

## Environment

```bash
conda create -n nano-megatron python=3.10 -y
conda activate nano-megatron
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Adjust the PyTorch CUDA wheel if your server uses a different CUDA runtime.

## Data

Prepare the tiny Shakespeare dataset with GPT-2 BPE tokenization:

```bash
python parallel/data/shakespeare/prepare.py
```

This creates:

```text
parallel/data/shakespeare/train.bin
parallel/data/shakespeare/val.bin
```

## TP Correctness Tests

Run these from the repository root. The examples below use two visible GPUs and `tensor_model_parallel_size=2`.

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_linear --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_mlp --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_attention --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_gpt --tensor-model-parallel-size 2
```

These tests compare the tensor-parallel modules against single-process reference implementations.

## TP Training

Small smoke test:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.pretrain_GPT \
  --tensor-model-parallel-size 2 \
  --data-dir parallel/data/shakespeare \
  --max-steps 20 \
  --log-interval 1 \
  --micro-batch-size 4 \
  --max-seq-len 64 \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 512 \
  --num-attention-heads 4 \
  --embedding-dropout 0.0 \
  --attention-dropout 0.0 \
  --residual-dropout 0.0 \
  --hidden-dropout 0.0
```

Larger TP run:

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.pretrain_GPT \
  --tensor-model-parallel-size 2 \
  --data-dir parallel/data/shakespeare \
  --max-steps 2000 \
  --log-interval 1 \
  --micro-batch-size 4 \
  --max-seq-len 64 \
  --num-layers 12 \
  --hidden-size 768 \
  --ffn-hidden-size 3072 \
  --num-attention-heads 12 \
  --embedding-dropout 0.0 \
  --attention-dropout 0.0 \
  --residual-dropout 0.0 \
  --hidden-dropout 0.0
```

## TP+DP Smoke Test

Use four GPUs with `TP=2` and `DP=2`:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 -m parallel.pretrain_GPT \
  --tensor-model-parallel-size 2 \
  --data-dir parallel/data/shakespeare \
  --max-steps 100 \
  --log-interval 1 \
  --micro-batch-size 4 \
  --max-seq-len 64 \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 512 \
  --num-attention-heads 4 \
  --embedding-dropout 0.0 \
  --attention-dropout 0.0 \
  --residual-dropout 0.0 \
  --hidden-dropout 0.0
```

This uses:

```text
world_size = 4
tensor_model_parallel_size = 2
data_parallel_size = 2
```

The current DP path uses PyTorch DDP over the data-parallel group. A stricter DP implementation should still add DP-rank-aware sampling and DP-averaged logging.

## Multi-Node Notes

For multi-node runs, all nodes must be able to connect to the master TCP port. If `ping` works but `nc -vz <master_ip> <port>` hangs, the issue is network/firewall policy rather than model code.

Static rendezvous example:

```bash
# node0
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=2 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=<node0_ip> \
  --master_port=<open_port> \
  -m parallel.tests.test_tp_linear \
  --tensor-model-parallel-size 2

# node1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --nnodes=2 \
  --node_rank=1 \
  --nproc_per_node=4 \
  --master_addr=<node0_ip> \
  --master_port=<open_port> \
  -m parallel.tests.test_tp_linear \
  --tensor-model-parallel-size 2
```

Useful checks:

```bash
ss -lntp | grep <open_port>
timeout 5 nc -vz <node0_ip> <open_port>
```

If the cluster only opens SSH ports, multi-node `torchrun` cannot rendezvous until an allowed TCP port is provided.
