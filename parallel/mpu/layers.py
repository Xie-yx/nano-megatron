import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from parallel.global_vars import get_args

from .mpu_initialize import get_tp_rank, get_tp_world_size
from .utils import divide
from .mappings import copy_to_tp_region, gather_from_tp_region, reduce_from_tp_region, scatter_to_tp_region


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
}

def set_tensor_model_parallel_attributes(tensor, is_parallel, dim):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)



def _initialize_affine_weight(
    weight,
    output_size,
    input_size,
    per_partition_size, 
    partition_dim,
    init_method,              
):
    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim
    )
    
    master_weight = torch.empty(
        output_size, 
        input_size, 
        dtype=weight.dtype, 
        requires_grad=False,
    )
    
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(args.params_dtype)
    
    # split and copy
    weight_list = torch.split(
        master_weight,
        per_partition_size,
        dim=partition_dim,
    )
    rank = get_tp_rank()
    
    cur_weight = weight_list[rank].contiguous()
    with torch.no_grad():
        weight.copy_(cur_weight)
    


# F.linear(input, weight, bias) = input @ weight.T + bias
# wetght 的一行对应 output 的一行

class ColumnParallelLinear(torch.nn.Module):
    
    def __init__(
        self, 
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        
        args = get_args()
        
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        world_size = get_tp_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                dtype = args.params_dtype,            
            )
        )
        _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.output_size_per_partition,
            partition_dim=0,
            init_method=init_method,
        )
        
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size_per_partition, 
                    dtype=args.params_dtype
                )
            )
            set_tensor_model_parallel_attributes(self.bias, True, dim=0)
            
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
        
    def forward(self, input_):
        input_parallel = copy_to_tp_region(input_)
        bias = self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias)

        if self.gather_output:
            output = gather_from_tp_region(output_parallel)
        else:
            output = output_parallel
            
        return output


class RowParallelLinear(torch.nn.Module):
    
    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        
        args = get_args()
        
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        world_size = get_tp_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                dtype=args.params_dtype,
            )
        )
        _initialize_affine_weight(
            self.weight,
            self.output_size,
            self.input_size,
            self.input_size_per_partition,
            partition_dim=1,
            init_method=init_method, 
        )
        
        if bias:
            self.bias = Parameter(
                torch.empty(
                    self.output_size,
                    dtype=args.params_dtype,
                )
            )
            
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)
            
    def forward(self, input_):
        input_parallel = input_
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tp_region(input_)
        
        output_parallel = F.linear(input_parallel, self.weight)
        
        output = reduce_from_tp_region(output_parallel)
        output = output + self.bias if self.bias is not None else output
        
        return output



class VocabParallelEmbedding(torch.nn.Module):
    
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        
        
    def forward(self, input_):
        pass
        
        