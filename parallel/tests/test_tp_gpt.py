import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.global_vars import get_args
from parallel.initialize import initialize_megatron
from parallel import mpu
from parallel.model.GPT_model import ParallelGPT


TEST_SEED = 4040
ATOL = 1e-5
RTOL = 1e-5


def _device() -> torch.device:
    return torch.device(get_args().device)


def _rank() -> int:
    return torch.distributed.get_rank()


def _world_size() -> int:
    return torch.distributed.get_world_size()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch, max diff={diff}")


class ReferenceMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int, residual_dropout: float, use_bias: bool):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, ffn_hidden_size, bias=use_bias)
        self.gelu = nn.GELU()
        self.dense_4h_to_h = nn.Linear(ffn_hidden_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x


class ReferenceSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attention_dropout: float, residual_dropout: float, use_bias: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        self.norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_size = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.norm_factor
        scores = scores - attention_mask * 10000.0
        probs = self.softmax(scores)
        probs = self.attention_dropout(probs)

        output = probs @ v
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, hidden_size)
        output = self.residual_dropout(self.out_proj(output))
        return output


class ReferenceTransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_hidden_size: int, attention_dropout: float, residual_dropout: float, layernorm_epsilon: float, use_bias: bool):
        super().__init__()
        self.pre_attn_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.attn = ReferenceSelfAttention(hidden_size, num_heads, attention_dropout, residual_dropout, use_bias)
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.mlp = ReferenceMLP(hidden_size, ffn_hidden_size, residual_dropout, use_bias)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = hidden_states + self.attn(self.pre_attn_layernorm(hidden_states), attention_mask)
        output = output + self.mlp(self.pre_mlp_layernorm(output))
        return output


class ReferenceGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.padded_vocab_size, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.embedding_dropout = nn.Dropout(args.hidden_dropout)
        self.blocks = nn.ModuleList(
            [
                ReferenceTransformerBlock(
                    hidden_size=args.hidden_size,
                    num_heads=args.num_attention_heads,
                    ffn_hidden_size=args.ffn_hidden_size,
                    attention_dropout=args.attention_dropout,
                    residual_dropout=args.residual_dropout,
                    layernorm_epsilon=args.layernorm_epsilon,
                    use_bias=args.use_bias,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        return F.linear(hidden_states, self.word_embeddings.weight)


def _build_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    torch.manual_seed(TEST_SEED)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return input_ids, position_ids, labels


def _build_causal_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)


def _copy_reference_to_parallel(ref_model: ReferenceGPT, tp_model: ParallelGPT) -> None:
    args = get_args()
    tp_rank = mpu.get_tp_rank()

    vocab_shard = tp_model.language_model.embedding.word_embeddings.num_embeddings_per_partition
    vocab_start = tp_rank * vocab_shard
    vocab_end = vocab_start + vocab_shard

    with torch.no_grad():
        tp_model.language_model.embedding.word_embeddings.weight.copy_(
            ref_model.word_embeddings.weight[vocab_start:vocab_end].to(
                tp_model.language_model.embedding.word_embeddings.weight.device
            )
        )
        tp_model.language_model.embedding.position_embeddings.weight.copy_(
            ref_model.position_embeddings.weight.to(
                tp_model.language_model.embedding.position_embeddings.weight.device
            )
        )
        tp_model.language_model.transformer.layer_norm.load_state_dict(ref_model.layer_norm.state_dict())

        for ref_block, tp_block in zip(ref_model.blocks, tp_model.language_model.transformer.blocks):
            tp_block.pre_attn_layernorm.load_state_dict(ref_block.pre_attn_layernorm.state_dict())
            tp_block.pre_mlp_layernorm.load_state_dict(ref_block.pre_mlp_layernorm.state_dict())

            qkv_shard = tp_block.attn.q_proj.output_size_per_partition
            qkv_start = tp_rank * qkv_shard
            qkv_end = qkv_start + qkv_shard

            tp_block.attn.q_proj.weight.copy_(ref_block.attn.q_proj.weight[qkv_start:qkv_end].to(tp_block.attn.q_proj.weight.device))
            tp_block.attn.k_proj.weight.copy_(ref_block.attn.k_proj.weight[qkv_start:qkv_end].to(tp_block.attn.k_proj.weight.device))
            tp_block.attn.v_proj.weight.copy_(ref_block.attn.v_proj.weight[qkv_start:qkv_end].to(tp_block.attn.v_proj.weight.device))
            if args.use_bias:
                tp_block.attn.q_proj.bias.copy_(ref_block.attn.q_proj.bias[qkv_start:qkv_end].to(tp_block.attn.q_proj.bias.device))
                tp_block.attn.k_proj.bias.copy_(ref_block.attn.k_proj.bias[qkv_start:qkv_end].to(tp_block.attn.k_proj.bias.device))
                tp_block.attn.v_proj.bias.copy_(ref_block.attn.v_proj.bias[qkv_start:qkv_end].to(tp_block.attn.v_proj.bias.device))

            out_shard = tp_block.attn.out_proj.input_size_per_partition
            out_start = tp_rank * out_shard
            out_end = out_start + out_shard
            tp_block.attn.out_proj.weight.copy_(ref_block.attn.out_proj.weight[:, out_start:out_end].to(tp_block.attn.out_proj.weight.device))
            if args.use_bias:
                tp_block.attn.out_proj.bias.copy_(ref_block.attn.out_proj.bias.to(tp_block.attn.out_proj.bias.device))

            fc1_shard = tp_block.mlp.dense_h_to_4h.output_size_per_partition
            fc1_start = tp_rank * fc1_shard
            fc1_end = fc1_start + fc1_shard
            tp_block.mlp.dense_h_to_4h.weight.copy_(ref_block.mlp.dense_h_to_4h.weight[fc1_start:fc1_end].to(tp_block.mlp.dense_h_to_4h.weight.device))
            if args.use_bias:
                tp_block.mlp.dense_h_to_4h.bias.copy_(ref_block.mlp.dense_h_to_4h.bias[fc1_start:fc1_end].to(tp_block.mlp.dense_h_to_4h.bias.device))

            fc2_shard = tp_block.mlp.dense_4h_to_h.input_size_per_partition
            fc2_start = tp_rank * fc2_shard
            fc2_end = fc2_start + fc2_shard
            tp_block.mlp.dense_4h_to_h.weight.copy_(ref_block.mlp.dense_4h_to_h.weight[:, fc2_start:fc2_end].to(tp_block.mlp.dense_4h_to_h.weight.device))
            if args.use_bias:
                tp_block.mlp.dense_4h_to_h.bias.copy_(ref_block.mlp.dense_4h_to_h.bias.to(tp_block.mlp.dense_4h_to_h.bias.device))


def test_parallel_gpt_forward_backward() -> None:
    args = get_args()
    batch_size = 2
    seq_len = min(8, args.max_position_embeddings)
    device = _device()

    torch.manual_seed(TEST_SEED + 1)
    ref_model = ReferenceGPT(args).to(device)
    tp_model = ParallelGPT(parallel_output=False).to(device)
    _copy_reference_to_parallel(ref_model, tp_model)

    ref_model.eval()
    tp_model.eval()

    input_ids, position_ids, labels = _build_batch(batch_size, seq_len, args.vocab_size, device)
    attention_mask = _build_causal_mask(batch_size, seq_len, device)

    logits_ref = ref_model(input_ids, position_ids, attention_mask)
    logits_tp = tp_model(input_ids, position_ids, attention_mask)
    _assert_close("gpt forward logits", logits_tp, logits_ref)

    loss_ref = F.cross_entropy(logits_ref.view(-1, logits_ref.size(-1)), labels.view(-1))
    loss_tp = F.cross_entropy(logits_tp.view(-1, logits_tp.size(-1)), labels.view(-1))
    loss_ref.backward()
    loss_tp.backward()

    vocab_shard = tp_model.language_model.embedding.word_embeddings.num_embeddings_per_partition
    vocab_start = mpu.get_tp_rank() * vocab_shard
    vocab_end = vocab_start + vocab_shard
    _assert_close(
        "word embedding grad",
        tp_model.language_model.embedding.word_embeddings.weight.grad,
        ref_model.word_embeddings.weight.grad[vocab_start:vocab_end],
    )

    ref_block = ref_model.blocks[0]
    tp_block = tp_model.language_model.transformer.blocks[0]

    qkv_shard = tp_block.attn.q_proj.output_size_per_partition
    qkv_start = mpu.get_tp_rank() * qkv_shard
    qkv_end = qkv_start + qkv_shard
    _assert_close("gpt q_proj grad", tp_block.attn.q_proj.weight.grad, ref_block.attn.q_proj.weight.grad[qkv_start:qkv_end])

    fc2_shard = tp_block.mlp.dense_4h_to_h.input_size_per_partition
    fc2_start = mpu.get_tp_rank() * fc2_shard
    fc2_end = fc2_start + fc2_shard
    _assert_close("gpt mlp output grad", tp_block.mlp.dense_4h_to_h.weight.grad, ref_block.mlp.dense_4h_to_h.weight.grad[:, fc2_start:fc2_end])

    if _rank() == 0:
        print("ParallelGPT forward/backward reference test passed.", flush=True)


def test_parallel_gpt_parallel_output_shape() -> None:
    args = get_args()
    batch_size = 2
    seq_len = min(8, args.max_position_embeddings)
    device = _device()

    model = ParallelGPT(parallel_output=True).to(device)
    model.eval()

    input_ids, position_ids, _ = _build_batch(batch_size, seq_len, args.vocab_size, device)
    attention_mask = _build_causal_mask(batch_size, seq_len, device)
    logits_parallel = model(input_ids, position_ids, attention_mask)

    expected_shape = (
        batch_size,
        seq_len,
        args.padded_vocab_size // args.tensor_model_parallel_size,
    )
    if logits_parallel.shape != expected_shape:
        raise AssertionError(
            f"parallel logits shape mismatch: got {tuple(logits_parallel.shape)}, expected {expected_shape}"
        )

    if _rank() == 0:
        print("ParallelGPT parallel_output shape test passed.", flush=True)


def main() -> None:
    initialize_megatron(
        args_defaults={
            "num_layers": 2,
            "hidden_size": 16,
            "ffn_hidden_size": 32,
            "num_attention_heads": 4,
            "vocab_size": 32,
            "max_seq_len": 8,
            "embedding_dropout": 0.0,
            "attention_dropout": 0.0,
            "residual_dropout": 0.0,
            "hidden_dropout": 0.0,
            "use_bias": True,
        }
    )
    args = get_args()

    if _world_size() != args.tensor_model_parallel_size:
        raise ValueError(
            "For this minimal test, world_size must equal tensor_model_parallel_size: "
            f"world_size={_world_size()}, tp={args.tensor_model_parallel_size}"
        )

    test_parallel_gpt_forward_backward()
    torch.distributed.barrier()
    test_parallel_gpt_parallel_output_shape()
    torch.distributed.barrier()

    if _rank() == 0:
        print("All ParallelGPT tests passed.", flush=True)


if __name__ == "__main__":
    main()
