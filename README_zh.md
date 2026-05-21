# nano-megatron 中文说明

`nano-megatron` 是一个用于学习 GPT 训练与 Megatron 风格并行技术的最小化项目。项目先实现并验证单卡 GPT baseline，再在 `parallel/` 中逐步实现 tensor parallel、data parallel、并行 embedding、并行 cross entropy 和分布式训练骨架。

当前项目重点不是追求训练效果，而是把大模型训练里的核心工程路径拆开验证：

- 单卡 GPT 训练基线
- TP 通信原语
- 列并行/行并行线性层
- TP MLP 和 TP Self-Attention
- VocabParallelEmbedding
- Vocab-parallel cross entropy
- ParallelGPT
- TP 训练闭环
- TP+DP 最小训练路径
- DP rank 感知的数据采样

## 目录结构

```text
nano-megatron/
  _single_gpu_baseline/     # 已验证的单卡 GPT baseline
  parallel/
    arguments.py            # Megatron 风格参数系统
    initialize.py           # 分布式与模型并行初始化
    training.py             # 简化版 CodeGeeX/Megatron 训练骨架
    pretrain_GPT.py         # GPT 预训练入口
    data/                   # tiny Shakespeare 数据准备
    model/                  # ParallelGPT 模型实现
    mpu/                    # TP group、通信原语、并行层、并行 loss
    tests/                  # TP 正确性测试
  requirements.txt
```

## 环境准备

```bash
conda create -n nano-megatron python=3.10 -y
conda activate nano-megatron
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

如果服务器 CUDA runtime 不是 CUDA 12.1，请换成对应的 PyTorch 安装命令。

## 数据准备

当前使用 tiny Shakespeare，并通过 GPT-2 BPE tokenizer 预处理。

```bash
python parallel/data/shakespeare/prepare.py
```

生成文件：

```text
parallel/data/shakespeare/train.bin
parallel/data/shakespeare/val.bin
```

## TP 正确性测试

下面命令需要在项目根目录执行。示例使用两张 GPU，并设置 `tensor_model_parallel_size=2`。

```bash
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_linear --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_mlp --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_attention --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.tests.test_tp_gpt --tensor-model-parallel-size 2
```

测试覆盖：

- `ColumnParallelLinear`
- `RowParallelLinear`
- `ParallelMLP`
- `ParallelSelfAttention`
- `ParallelGPT`

这些测试会和单进程 reference 实现进行前向/反向数值对照。

## TP 训练

最小 smoke test：

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

稍大的 TP 训练配置：

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

当前训练路径使用：

- `ParallelGPT(parallel_output=True)`
- `VocabParallelEmbedding`
- `vocab_parallel_cross_entropy`
- `torch.optim.AdamW`
- `tqdm` 进度条

## TP+DP Smoke Test

四张 GPU，`TP=2`，`DP=2`：

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

对应并行关系：

```text
world_size = 4
tensor_model_parallel_size = 2
data_parallel_size = 2
```

当前 DP 通过 PyTorch DDP 接入 `data_parallel_group`。数据采样已经做了 DP rank 感知：

- 同一个 TP group 内的 rank 使用相同 batch
- 不同 DP rank 使用不同 batch

后续如果要进一步完善 DP 语义，建议补充：

- DP 组内 loss 平均日志
- TP/DP checkpoint 保存与恢复

## 多机训练说明

多机训练需要确保节点之间可以通过 TCP 端口互通。`ping` 通不代表 `torchrun` 可用，必须确认 master 端口可连接：

```bash
timeout 5 nc -vz <node0_ip> <port>
```

如果服务器只开放 SSH 端口，`torchrun` 多机 rendezvous 会卡住。需要管理员开放可用 TCP 端口，或者在允许的网络环境中运行。

静态 rendezvous 示例：

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

常用排查命令：

```bash
ss -lntp | grep <open_port>
timeout 5 nc -vz <node0_ip> <open_port>
```

如果 `nc` 超时，问题在网络/防火墙/端口策略，不在模型代码。

## 当前状态

已完成：

- 单卡 baseline
- TP primitive
- TP linear
- TP MLP
- TP attention
- TP GPT 数值对照测试
- vocab-parallel cross entropy
- TP 训练闭环
- TP+DP smoke 路径
- DP rank 感知数据采样

后续可继续完善：

- DP loss 平均日志
- checkpoint 保存与恢复
- 多机网络环境下的实际训练验证
