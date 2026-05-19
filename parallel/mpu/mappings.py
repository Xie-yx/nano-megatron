import torch

from .mpu_initialize import get_tp_world_size, get_tp_group, get_tp_rank
from .utils import split_tensor_along_last_dim

def _reduce(input_):
    
    if get_tp_world_size() == 1:
        return input_
    
    torch.distributed.all_reduce(input_, group=get_tp_group())
    
    return input_


def _split(input_):
    world_size = get_tp_world_size()
    if world_size == 1:
        return input_
    
    rank = get_tp_rank()
    
    input_list = split_tensor_along_last_dim(input_, world_size)
    output = input_list[rank].contiguous()
    
    return output

def _gather(input_):
    
    world_size = get_tp_world_size()
    if world_size == 1:
        return input_
    
    rank = get_tp_rank()
    last_dim = input_.dim() - 1
    
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=get_tp_group()
    )
    output = torch.cat(tensor_list, dim=last_dim).contiguous()
    return output



class _CopyToTensorParallelRegion(torch.autograd.Function):
    
    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_
    
    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _GatherFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)
    
    
class _ScatterToTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _split(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)
    
    
class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# column parallelism linear f 算子
def copy_to_tp_region(input_):
    return _CopyToTensorParallelRegion.apply(input_)

# column parallelism linear g 算子
def gather_from_tp_region(input_):
    return _GatherFromTensorParallelRegion.apply(input_)

# row parallelism linear f 算子
def scatter_to_tp_region(input_):
    return _ScatterToTensorParallelRegion.apply(input_)

# row parallelism linear g 算子
def reduce_from_tp_region(input_):
    return _ReduceFromTensorParallelRegion.apply(input_)