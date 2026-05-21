import torch
import torch.nn as nn
from torch.nn import functional as F
import math

from parallel.global_vars import get_args
from parallel import mpu

from ..mpu.utils import init_method_normal, scaled_init_method_normal


class ParallelGPT(nn.Module):
    
    def __init__(
        self,
        parallel_output = False,
    ):
        super().__init__()
        args = get_args()
        self.parallel_output = parallel_output
        
        self.language_model, self.language_model_key = get_language_model(
            init_method = init_method_normal(args.init_method_std),
            scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers),  
        )
        
    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        labels = None,
        forward_method_parallel_output = None,
    ):
        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask
        )
        
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output
            
        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output
        )
        if labels is None:
            return output
        else:
            loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
            return loss
        
        
        
def parallel_lm_logits(
    input_,
    word_embeddings_weight,
    parallel_output,
    bias = None,
):
    input_parallel = mpu.copy_to_tp_region(input_.contiguous())
    
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)
    
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tp_region(logits_parallel.contiguous())


def get_language_model(
    init_method = None,
    scaled_init_method = None,
):
    args = get_args()
    
    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
        
    language_model = TransformerLanguageModel(
        init_method = init_method,
        output_layer_init_method = scaled_init_method,    
    )
    language_model_key = 'language_model'
    return language_model, language_model_key



class TransformerLanguageModel(nn.Module):
    
    def __init__(
        self,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()
        args = get_args()
        
        self.hidden_size = args.hidden_size
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        
        
        self.embedding = Embedding(
            hidden_size = args.hidden_size,
            vocab_size = args.padded_vocab_size,
            max_sequence_length = args.max_position_embeddings,
            embedding_dropout = args.hidden_dropout,
            init_method = init_method,
        )
        
        self._embedding_key = 'embedding'
        
        self.transformer = ParallelTransformer(
            init_method = init_method,
            output_layer_init_method = output_layer_init_method,
        )
        self._transformer_key = 'transformer'
        
        
    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
    ):
        embedding_output = self.embedding(input_ids, position_ids)
        
        transformer_output = self.transformer(
            embedding_output,
            attention_mask,
        )

        return transformer_output




class Embedding(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout,
        init_method=None,        
    ):
        super().__init__()
        args = get_args()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.init_method = init_method
        self.max_sequence_length = max_sequence_length
        
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size,
            hidden_size,
            init_method=init_method,
        )
        self._word_embeddings_key = 'word_embeddings'

        self.position_embeddings = nn.Embedding(
            max_sequence_length,
            hidden_size,
        )
        self._position_embeddings_key = 'position_embeddings'

        self.position_embeddings = self.position_embeddings.to(dtype=args.params_dtype)
        
        self.init_method(self.position_embeddings.weight)
        
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        
                
        
        
    def forward(
        self,
        input_ids,
        position_ids,
    ):
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = word_embeddings + position_embeddings
        embeddings = self.embedding_dropout(embeddings) 
        return embeddings
        
        

class ParallelTransformer(nn.Module):
    
    def __init__(
        self,
        init_method,
        output_layer_init_method,
    ):
        super().__init__()
        args = get_args()
    
        self.num_layers = args.num_layers
        
        def build_block(layer_number):
            return ParallelTransformerBlock(
                init_method = init_method,
                output_layer_init_method = output_layer_init_method,
                layer_number = layer_number,
            )
            
        self.blocks = nn.ModuleList(
            [build_block(i+1) for i in range(self.num_layers)]
        )
        
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)
     
     
    def forward(
        self,
        hidden_states,
        attention_mask,
    ):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.layer_norm(hidden_states)
        
        return hidden_states
    




class ParallelTransformerBlock(nn.Module):
    
    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
    ):
        super().__init__()
        args = get_args()
        self.pre_attn_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)
        
        self.attn = ParallelSelfAttention(
            init_method = init_method,
            output_layer_init_method = output_layer_init_method,
            layer_number = layer_number
        )
        
        self.pre_mlp_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.mlp = ParallelMLP(
            init_method = init_method,
            output_layer_init_method = output_layer_init_method,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        
    ):
        output = hidden_states + self.attn(self.pre_attn_layernorm(hidden_states), attention_mask)
        output = output + self.mlp(self.pre_mlp_layernorm(output))
        return output



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
            bias=args.use_bias,
            gather_output=False,
            init_method=init_method,
        )
        self.k_proj = mpu.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=args.use_bias,
            gather_output=False,
            init_method=init_method,
        )
        self.v_proj = mpu.ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=args.use_bias,
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
            bias=args.use_bias,
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
            bias=args.use_bias,
            gather_output=False,
            init_method=init_method
        )
        
        self.activation_func = nn.GELU()
        
        self.dense_4h_to_h = mpu.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size, 
            bias=args.use_bias,
            input_is_parallel=True, 
            init_method=output_layer_init_method
        )

        self.dropout = nn.Dropout(args.residual_dropout)

    
    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states



