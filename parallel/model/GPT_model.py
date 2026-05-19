import torch
import torch.nn as nn
import math

from parallel.global_vars import get_args
from parallel import mpu



class ParallelSelfAttention(nn.Module):

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
    ):
        super().__init__()
        args = get_args()
        # self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        self.layer_number = max(1, layer_number)
        
        self.hidden_size = args.hidden_size
        world_size = mpu.get_tp_world_size()
        self.hidden_size_per_partition = mpu.divide(args.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(args.hidden_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = mpu.divide(args.num_attention_heads, world_size)
        
        
        self.q_proj = mpu.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            gather_output=False,
            init_method=init_method,
        )
        self.k_proj = mpu.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            gather_output=False,
            init_method=init_method,
        )
        self.v_proj = mpu.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            gather_output=False,
            init_method=init_method,
        )
        
        self.norm_factor = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.attention_dropout = nn.Dropout(args.attention_dropout)
        self.residual_dropout = nn.Dropout(args.residual_dropout)
        
        self.out_proj = mpu.RowParallelLinear(
            args.hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
        )
        
    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        B, T, C = hidden_states.size()
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        
        new_shape = q.size()[:-1] + (
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head
            )
        q = q.view(*new_shape).transpose(1, 2)
        k = k.view(*new_shape).transpose(1, 2)
        v = v.view(*new_shape).transpose(1, 2)
        
        attention_scores = (q @ k.transpose(-2, -1)) * self.norm_factor
        attention_scores = attention_scores - attention_mask * 10000.0
        
        
        attention_probs = self.softmax(attention_scores)

        
        attention_probs = self.attention_dropout(attention_probs)
        
        output = attention_probs @ v
        output = output.transpose(1, 2).contiguous().view(B, T, self.hidden_size_per_partition)
        output = self.residual_dropout(self.out_proj(output))
        
        return output
        

class ParallelMLP(nn.Module):
    
    def __init__(
        self,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()
        args = get_args()
        
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            args.hidden_size, 
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method
        )
        
        self.activation_func = nn.GELU()
        
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size, 
            input_is_parallel=True, 
            init_method=output_layer_init_method
        )

    
    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states



