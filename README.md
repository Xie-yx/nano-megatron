

MLP 单元测试
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_mlp --tensor-model-parallel-size 2
```

attention 单元测试
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_attention --tensor-model-parallel-size 2
```

TP test
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_gpt --tensor-model-parallel-size 2
```

TP training smoke test with vocab-parallel cross entropy
```
python parallel/data/shakespeare/prepare.py
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

Full TP validation sequence
```
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_linear --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_mlp --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_attention --tensor-model-parallel-size 2
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 -m parallel.test_tp_gpt --tensor-model-parallel-size 2
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


TP+DP
```
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

