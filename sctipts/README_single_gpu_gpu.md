# Single GPU Training on Server

这份说明面向 **Linux GPU 服务器**，目标是在单卡上运行当前项目的 `tinyshakespeare -> prepare.py -> train_single_gpu.py` 流程。

## 1. 环境准备

推荐使用单独的 `conda` 环境。

## 1.0 推荐版本基线

为了兼顾：

- 当前 toy 单卡训练稳定可用
- 后续向 Megatron 风格并行训练扩展时尽量少踩环境坑

建议优先采用下面这套 **保守且相对稳妥** 的版本基线：

- `Python==3.10.*`
- `PyTorch==2.1.2`
- `CUDA==12.1`（次选 `11.8`）
- `numpy==1.26.4`
- `requests==2.31.0`
- `tiktoken==0.5.2`

这套版本选择的出发点是：

- 比当前很多“能跑就行”的最新版本更保守
- 比 `CodeGeeX` 那种宽版本约束更容易复现
- 对未来接入 `flash-attn`、`deepspeed`、Apex 或 TP 改造更友好

### 1.1 创建 Python 环境

```bash
conda create -n nano-megatron python=3.10 -y
conda activate nano-megatron
```

### 1.2 安装 PyTorch（GPU 版）

先在服务器上确认 CUDA 驱动信息：

```bash
nvidia-smi
```

然后安装与你服务器 CUDA 驱动兼容的 PyTorch。下面给出两个常见示例。

#### 方案 A：CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

#### 方案 B：CUDA 11.8

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```

如果你不确定该选哪个版本，优先参考 PyTorch 官方安装页面，并选择与服务器驱动兼容的版本。

### 1.3 安装项目当前所需依赖

```bash
pip install -r requirements.txt
```

如果你暂时不想通过 `requirements.txt` 安装，也可以直接执行：

```bash
pip install numpy==1.26.4 requests==2.31.0 tiktoken==0.5.2
```

### 1.4 验证 GPU 可用

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

如果输出中 `torch.cuda.is_available()` 为 `True`，说明 GPU 环境可用。

### 1.5 未来并行扩展的依赖建议

当前 toy 项目 **不需要现在就安装** 下面这些包，但如果后续开始做 TP / Megatron 风格并行训练，可以参考以下策略：

- `flash-attn`：建议在服务器上根据实际 `torch/cuda` 版本安装兼容版本
- `deepspeed`：建议在服务器上现装，不建议在本地 CPU 环境提前锁死
- `apex`：建议按 NVIDIA 官方源码与服务器环境现编译

也就是说：

- **当前核心依赖锁死版本**
- **未来并行依赖在 GPU 服务器上按环境安装**

这是比现在就把所有并行依赖都写死到一个文件里更稳妥的做法。

## 2. 获取代码

```bash
git clone <your-repo-url>
cd nano-megatron
```

如果代码已经在服务器上，可以直接进入仓库目录。

## 3. 准备数据

当前单卡训练流程使用 `tinyshakespeare`，由 `data/shakespeare/prepare.py` 自动下载并生成：

- `data/shakespeare/input.txt`
- `data/shakespeare/train.bin`
- `data/shakespeare/val.bin`

执行：

```bash
python data/shakespeare/prepare.py
```

## 4. 启动单卡 GPU 训练

### 4.1 最小训练命令

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/shakespeare_run \
  --csv-log-name loss_log.csv \
  --max-steps 200 \
  --eval-interval 50 \
  --save-interval 100
```

这条命令会：

- 从 `data/shakespeare/train.bin` 和 `val.bin` 读取数据
- 在 `out/shakespeare_run/` 下保存 checkpoint
- 在 `out/shakespeare_run/loss_log.csv` 中保存训练与评估日志

### 4.2 建议的较小起步配置

如果你想先快速验证 GPU 训练链路是否正常，可以先用较小模型：

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/shakespeare_small \
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

如果你希望先做一个更轻量的 GPU 冒烟训练，也可以把训练步数压到很小：

```bash
python -u sctipts/train_single_gpu.py \
  --data-dir data/shakespeare \
  --device cuda \
  --out-dir out/shakespeare_smoke \
  --batch-size 4 \
  --max-steps 20 \
  --eval-interval 10 \
  --save-interval 0 \
  --max-seq-len 64 \
  --num-layers 2 \
  --hidden-size 128 \
  --ffn-hidden-size 512 \
  --num-attention-heads 4
```

如果显存不足，可以优先降低：

- `--batch-size`
- `--max-seq-len`
- `--hidden-size`
- `--num-layers`

## 5. 当前训练输出

训练过程中，脚本会打印：

- `train loss`
- `eval train loss`
- `eval val loss`
- checkpoint 保存路径
- CSV 日志保存路径

默认 checkpoint 保存到：

- `out/checkpoint_step_xxxxxx.pt`

如果你在命令里显式指定了 `--out-dir out/shakespeare_run`，则 checkpoint 会保存到：

- `out/shakespeare_run/checkpoint_step_xxxxxx.pt`

默认 CSV 日志保存到：

- `out/loss_log.csv`

如果你在命令里显式指定了 `--out-dir out/shakespeare_run`，则 CSV 会保存到：

- `out/shakespeare_run/loss_log.csv`

CSV 中会记录：

- `event`：`train` 或 `eval`
- `step`
- `train_loss`
- `eval_train_loss`
- `eval_val_loss`
- `learning_rate`
- `step_time_ms`

## 6. 如何简单评估训练效果

当前实现下，**最简单、最可靠的评估方式是看验证集 loss**。

### 6.1 优先看 `eval val loss`

这是当前阶段最重要的指标。

判断标准：

- `eval val loss` 持续下降：训练有效
- `train loss` 降、`val loss` 不降：可能开始过拟合
- `train loss` 和 `val loss` 都几乎不动：学习率、模型规模或训练步数可能需要调整

### 6.2 看 `train loss` 和 `val loss` 的差距

经验上：

- 两者都下降且差距不大：正常
- `train loss` 很低但 `val loss` 明显更高：过拟合迹象

### 6.3 可以顺手看 perplexity

语言建模里常用：

$$
\text{ppl} = e^{\text{val loss}}
$$

例如：

- `val loss = 3.0`，则 `ppl \approx e^3 \approx 20.1`
- `val loss = 2.5`，则 `ppl \approx 12.2`

在相同数据和 tokenizer 下，**perplexity 越低越好**。

### 6.4 现阶段最推荐的评估流程

对于当前项目，建议每次主要比较：

- 相同步数下的 `eval val loss`
- 不同模型配置下的 `eval val loss`
- 不同 checkpoint 的 `eval val loss`

也就是说，当前最实用的判断方法是：

1. 固定数据集
2. 固定 tokenizer
3. 改模型配置或训练步数
4. 比较 `val loss`

## 7. 训练效果的简单判断标准

以当前阶段为目标，可以这样判断：

- **链路正确**：loss 能正常下降
- **训练有效**：`eval val loss` 在多个评估点上整体下降
- **配置可用**：没有频繁 OOM，训练速度可接受
- **值得继续放大**：小模型在 `tinyshakespeare` 上能稳定收敛

## 8. 当前阶段不必过早做的事情

在服务器单卡阶段，先不用急着做：

- 分布式训练
- 混合精度复杂优化
- 大规模语料
- 复杂 benchmark
- 复杂任务评估集

当前最重要的是：

- 单卡 GPU 训练稳定运行
- 数据准备流程稳定
- checkpoint 可保存
- `val loss` 能作为稳定指标

## 9. 下一步建议

当单卡 GPU 训练稳定后，推荐按这个顺序继续：

1. 补 `resume checkpoint`
2. 补更规范的学习率调度
3. 增加生成函数，用固定 prompt 做定性观察
4. 再进入 TP / Megatron 风格并行改造

## 10. 为什么采用这套版本

参考项目里：

- `CodeGeeX` 的依赖范围较宽，例如 `torch>=1.10.0`
- `Megatron-LM` 更强调使用稳定一致的 NVIDIA PyTorch / CUDA 环境

对你这个项目来说，更推荐采用中间路线：

- 不追最新版本
- 不放太宽的版本范围
- 先固定一套稳定可复现的基础环境

因此，当前推荐：

- **核心依赖写具体版本**
- **并行依赖放到后续 GPU 服务器阶段再安装**
