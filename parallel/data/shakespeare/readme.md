
# tiny shakespeare

Tiny shakespeare, of the good old char-rnn fame :)

使用方式：

1. 运行 `prepare.py` 下载 `input.txt` 并生成 `train.bin` 与 `val.bin`
2. 运行 `sctipts/train_single_gpu.py --data-dir data/shakespeare`

After running `prepare.py`:

- train.bin has 301,966 tokens
- val.bin has 36,059 tokens
