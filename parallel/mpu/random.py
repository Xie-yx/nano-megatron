import torch

from .mpu_initialize import get_tp_rank


_MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"


class CudaRNGStatesTracker:
    """Minimal CUDA RNG tracker for tensor-parallel regions."""

    def __init__(self):
        self.states_ = {}
        self.seeds_ = set()

    def reset(self):
        self.states_ = {}
        self.seeds_ = set()

    def add(self, name, seed):
        if seed in self.seeds_:
            raise RuntimeError(f"seed {seed} already exists")
        if name in self.states_:
            raise RuntimeError(f"cuda rng state {name} already exists")

        self.seeds_.add(seed)
        orig_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(orig_rng_state)


_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed: int) -> None:
    """Initialize CUDA RNG for tensor-parallel execution.

    Minimal Megatron-style behavior:
    - default CUDA RNG uses the data-parallel seed
    - tensor-parallel tracker uses a rank-shifted seed
    """
    if seed <= 0:
        raise ValueError(f"seed ({seed}) should be a positive integer.")

    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tp_rank()
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    torch.cuda.manual_seed(data_parallel_seed)
    _CUDA_RNG_STATE_TRACKER.add(
        _MODEL_PARALLEL_RNG_TRACKER_NAME,
        tensor_model_parallel_seed,
    )
