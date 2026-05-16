# Single GPU Training

最小单卡训练入口是 `sctipts/train_single_gpu.py`。

当前流程使用 `data/shakespeare/prepare.py` 先把 `tinyshakespeare` 预处理成 `train.bin` 和 `val.bin`。

训练脚本会：
- 读取 `train.bin` / `val.bin`
- 使用 GPT-2 BPE token id 直接采样 `input_ids` 和 `labels`
- 训练 `model/GPT_model.py` 中的 GPT
- 定期评估并保存 checkpoint

先准备数据：

```powershell
E:/anaconda3/envs/nano-megatron/python.exe data/shakespeare/prepare.py
```

示例：

```powershell
E:/anaconda3/envs/nano-megatron/python.exe sctipts/train_single_gpu.py `
  --data-dir data/shakespeare `
  --max-steps 20 `
  --batch-size 4 `
  --max-seq-len 32 `
  --num-layers 2 `
  --hidden-size 64 `
  --ffn-hidden-size 256 `
  --num-attention-heads 4 `
  --eval-interval 10 `
  --save-interval 20
```
 
说明：
- `data/shakespeare/prepare.py` 默认使用 `tiktoken` 的 `gpt2` 编码。
- 因此训练脚本默认 `vocab_size=50257`。
