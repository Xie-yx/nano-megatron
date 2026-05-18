import os
import datetime

import numpy as np
import random
import torch
from .global_vars import set_global_variables, get_args


from . import mpu




def initialize_megatron(
    extra_args_provider=None,
    args_defaults: dict[str, object] | None = None,
    ignore_unknown_args: bool = False,
):
    """Initialize the model parallel environment."""
    assert torch.cuda.is_available(), "Megatron requires CUDA."
    set_global_variables(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        ignore_unknown_args=ignore_unknown_args,
    )
    args = get_args()
    
    _initialize_distributed()
    
    _set_random_seed(args.seed)
   



def _initialize_distributed():
    _initialize_torch_distributed()
    
    args = get_args()
    
    mpu.initialize_model_parallel(
        args.tensor_model_parallel_size,
        args.pipeline_model_parallel_size,
    )

    

def _initialize_torch_distributed() -> None:
    args = get_args()
    
    device_count = torch.cuda.device_count()
    
    if torch.distributed.is_initialized():
        
        if args.rank==0:
            print(
                "torch.distributed is already initialized, skipping initialization.",
                flush=True,
            )
            
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        return
     
    if args.rank == 0:
        print("> initializing torch distributed ...", flush=True)

    if device_count > 0:
        device = args.rank % device_count
        if args.local_rank is not None:
            assert (
                args.local_rank == device
            ), "expected local-rank to be the same as rank % device-count."
        else:
            args.local_rank = device
        
        torch.cuda.set_device(device)
    
    # todo: support torchrun
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "29500")
    init_method += f"{master_ip}:{master_port}"
    print(
        f"  > (rank={args.rank}) initializing process group: "
        f"world_size={args.world_size} "
        f"backend={args.distributed_backend} "
        f"init_method={init_method}",
        flush=True,
    )
    timeout = datetime.timedelta(minutes=args.dist_timeout)
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        init_method=init_method,
        timeout=timeout,
    )
    print(f"  > (rank={args.rank}) process group initialized")

    
    


def _set_random_seed(seed_):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pp_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            mpu.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))
