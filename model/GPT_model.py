import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass
import math


@dataclass
class GPTConfig:
    num_layers: int = 12
    hidden_size: int = 768
    ffn_hidden_size: int = 3072
    num_heads: int = 12
    vocab_size: int = 50257
    max_seq_len: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    use_bias: bool = True
    layernorm_epsilon: float = 1e-5



class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.token_embedding.weight = self.lm_head.weight # 共享 Embedding 权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        device = input_ids.device
        b, t = input_ids.size()
        assert t <= self.config.max_seq_len, "输入序列长度超过模型的最大序列长度"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        token_emb = self.token_embedding(input_ids) # (b, t, hidden_size)
        pos_emb = self.position_embedding(pos) # (t, hidden_size)
        
        x = self.dropout(token_emb + pos_emb)
        for block in self.blocks:
            x = block(x)        
        x = self.ln_f(x)

        
        logits = self.lm_head(x) # (b, t, vocab_size)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        else:
            loss = None
            
        return logits, loss
        
        

class TransformerBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_heads == 0, "hidden_size 必须是 num_heads 的整数倍"
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_bias)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_bias)
        

        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                             .view(1, 1, config.max_seq_len, config.max_seq_len))
        
        self.proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.use_bias)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, hidden size
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, num_heads, T, head_size)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, num_heads, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1) # (B, num_heads, T, T)
        attn = self.attn_dropout(attn)
        y = attn @ v # (B, num_heads, T, head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=config.use_bias)
        self.gelu = nn.GELU()
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.residual_dropout)
        
        
    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.gelu(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x
    





    
    
    
    



