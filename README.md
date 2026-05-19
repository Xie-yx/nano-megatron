

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