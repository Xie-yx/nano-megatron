import torch


from .utils import ensure_divisibility, divide


_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None



def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True



def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
):
    if model_parallel_is_initialized():
        print(
            "tensor model parallel is already initialized, skipping initialization.",
            flush=True,
        )
        return
    
    assert pipeline_model_parallel_size == 1, "Pipeline model parallelism is not supported yet."
    
    assert torch.distributed.is_initialized(), "torch.distributed must be initialized first."
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = min(tensor_model_parallel_size, world_size)
    pipeline_model_parallel_size = min(pipeline_model_parallel_size, world_size)
    data_parallel_size = divide(world_size, tensor_model_parallel_size * pipeline_model_parallel_size)

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    num_data_parallel_groups = world_size // data_parallel_size
    
    rank = torch.distributed.get_rank()
    

    # TP
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            
    
    # PP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, "pipeline model parallel group is already initialized"
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group

    # DP
    global _DATA_PARALLEL_GROUP
    assert _DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
    for i in range(pipeline_model_parallel_size):
        st = i * num_pipeline_model_parallel_groups
        ed = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(st+j, ed, tensor_model_parallel_size)    
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    

def get_tp_group():
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is not None
    ), "tp group is not initialized"
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_tp_world_size():
    return torch.distributed.get_world_size(group=get_tp_group())



def get_tp_rank():
    return torch.distributed.get_rank(group=get_tp_group())



def get_dp_group():
    assert (
        _DATA_PARALLEL_GROUP is not None
    ), "dp group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_dp_world_size():
    return torch.distributed.get_world_size(group=get_dp_group())


def get_dp_rank():
    return torch.distributed.get_rank(group=get_dp_group())


def get_pp_group():
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), "pp group is not initialized"
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_pp_world_size():
    return torch.distributed.get_world_size(group=get_pp_group())


def get_pp_rank():
    return torch.distributed.get_rank(group=get_pp_group())



def destroy_model_parallel():
    """Destroy the model parallel environment."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
