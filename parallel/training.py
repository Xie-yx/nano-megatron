import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from .initialize import initialize_megatron

from .global_vars import get_args

from . import mpu




def pretrain(
    model_provider,
    train_valid_test_dataset_provider,
    forward_step_func,
    extra_args_provider=None,
    args_defaults={},
):
    
    initialize_megatron(
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
    )
    
    args = get_args()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)


    (
        train_data_iterator,
        valid_data_iterator,
        test_data_iterator,
    ) = build_train_valid_test_data_iterators(train_valid_test_dataset_provider)


    return train(
        forward_step_func,
        model,
        optimizer,
        lr_scheduler,
        train_data_iterator,
        valid_data_iterator,
    )
    
    
def unwrap_model(model):
    if isinstance(model, list):
        modules = model
    else:
        modules = [model]

    unwrapped = []
    for module in modules:
        if hasattr(module, "module"):
            unwrapped.append(module.module)
        else:
            unwrapped.append(module)
    return unwrapped 
    

def setup_model_and_optimizer(model_provider):
    args = get_args()
    model = get_model(model_provider)
    
    optimizer = get_optimizer(model)
    lr_scheduler = get_lr_scheduler(optimizer)
    
    args.iteration = 0
    
    return model, optimizer, lr_scheduler
    
    
def get_model(model_provider):
    model = model_provider(
        pre_process = True,
        post_process = True,
    )

    if not isinstance(model, list):
        model = [model]
        
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)
    
    for model_module in model:
        model_module.cuda(torch.cuda.current_device())
        
        
    i = torch.cuda.current_device()
    model = [
        torchDDP(
            model_module,
            device_ids=[i],
            output_device=i,
            process_group=mpu.get_dp_group(),
        )
        for model_module in model
    ]
    return model
         

def get_optimizer(model):
    args = get_args()
    unwrapped_model = unwrap_model(model)
    param_groups = get_param_groups(unwrapped_model)
    
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return optimizer
    

def get_param_groups(modules):
    decay_params = []
    no_decay_params = []

    for module in modules:
        for submodule in module.modules():
            for name, param in submodule.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                if isinstance(submodule, nn.LayerNorm) or name == "bias":
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

    return [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_lr_scheduler(optimizer):
    args = get_args()
    
    total_steps = args.max_steps
    warmup_steps = min(10, max(1, total_steps // 100))
    min_lr = 0.0

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_scale = min_lr / args.learning_rate
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _TokenDataIterator:
    def __init__(self, tokens):
        self.tokens = tokens

    def __iter__(self):
        return self

    def __next__(self):
        return self.tokens


def _resolve_token_dtype():
    args = get_args()
    token_dtype = getattr(args, "token_dtype", None)
    if token_dtype is None:
        padded_vocab_size = getattr(args, "padded_vocab_size", args.vocab_size)
        return np.uint16 if padded_vocab_size <= np.iinfo(np.uint16).max else np.uint32
    if token_dtype in (np.uint16, "uint16"):
        return np.uint16
    if token_dtype in (np.uint32, "uint32"):
        return np.uint32
    if token_dtype in (np.int32, "int32"):
        return np.int32
    raise ValueError(f"Unsupported token dtype: {token_dtype}")


def build_train_valid_test_data_iterators(train_valid_test_dataset_provider):
    args = get_args()
    token_dtype = _resolve_token_dtype()

    train_ds, valid_ds, test_ds = None, None, None
    if train_valid_test_dataset_provider is not None:
        datasets = train_valid_test_dataset_provider()
        if datasets is None:
            datasets = (None, None, None)
        if len(datasets) != 3:
            raise ValueError(
                "train_valid_test_dataset_provider must return (train_ds, valid_ds, test_ds)"
            )
        train_ds, valid_ds, test_ds = datasets

    def resolve_data_file(filename):
        data_dir = args.data_dir
        candidate = os.path.join(data_dir, filename)
        if os.path.exists(candidate):
            return candidate

        if not os.path.isabs(data_dir):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(repo_root, data_dir, filename)
            if os.path.exists(candidate):
                return candidate

        return os.path.join(data_dir, filename)

    if train_ds is None:
        train_path = resolve_data_file("train.bin")
        if os.path.exists(train_path):
            train_ds = np.memmap(train_path, dtype=token_dtype, mode="r")

    if valid_ds is None:
        valid_path = resolve_data_file("val.bin")
        if os.path.exists(valid_path):
            valid_ds = np.memmap(valid_path, dtype=token_dtype, mode="r")

    if test_ds is None:
        test_path = resolve_data_file("test.bin")
        if os.path.exists(test_path):
            test_ds = np.memmap(test_path, dtype=token_dtype, mode="r")

    args.do_train = train_ds is not None
    args.do_valid = valid_ds is not None
    args.do_test = test_ds is not None

    train_data_iterator = _TokenDataIterator(train_ds) if train_ds is not None else None
    valid_data_iterator = _TokenDataIterator(valid_ds) if valid_ds is not None else None
    test_data_iterator = _TokenDataIterator(test_ds) if test_ds is not None else None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def forward_backward_no_pipeline(
    forward_step_func,
    data_iterator,
    model,
    forward_only=False,
):
    assert len(model) == 1, "Model list should have only one model for this training loop"
    model = model[0]
    output_tensor, loss_func = forward_step_func(data_iterator, model)
    loss, loss_dict = loss_func(output_tensor)
    
    if not forward_only:
        loss.backward()
        
    return [loss_dict]


def train_step(
    forward_step_func,
    data_iterator,
    model,
    optimizer,
    lr_scheduler,    
):
   
    args = get_args()

    optimizer.zero_grad(set_to_none=True)

    loss_dicts = forward_backward_no_pipeline(
        forward_step_func,
        data_iterator,
        model,
        forward_only=False,
    )
    
    if args.grad_clip is not None and args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for model_module in model for p in model_module.parameters()],
            args.grad_clip,
        )
        
    optimizer.step()
    
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return loss_dicts[0]
    
    

def train(
    forward_step_func,
    model,
    optimizer,
    lr_scheduler,
    train_data_iterator,
    valid_data_iterator=None,
):
    args = get_args()

    if train_data_iterator is None:
        raise RuntimeError(
            "train_data_iterator is None; no training data was built. "
            f"Check --data-dir={args.data_dir!r} from cwd={os.getcwd()!r}."
        )

    for model_module in model:
        model_module.train()

    iteration = getattr(args, "iteration", 0)
    is_log_rank = torch.distributed.get_rank() == 0
    progress = None
    if is_log_rank and tqdm is not None:
        progress = tqdm(
            total=args.max_steps,
            initial=iteration,
            desc="training",
            dynamic_ncols=True,
        )

    while iteration < args.max_steps:
        loss_dict = train_step(
            forward_step_func,
            train_data_iterator,
            model,
            optimizer,
            lr_scheduler,
        )

        iteration += 1
        args.iteration = iteration

        if is_log_rank:
            lr = optimizer.param_groups[0]["lr"]
            loss_values = {}
            for name, value in loss_dict.items():
                if torch.is_tensor(value):
                    value = value.detach().float().item()
                loss_values[name] = value

            if progress is not None:
                progress.update(1)
                if iteration % args.log_interval == 0:
                    progress.set_postfix(
                        {"lr": f"{lr:.3e}", **{k: f"{v:.4f}" for k, v in loss_values.items()}}
                    )
            elif iteration % args.log_interval == 0:
                loss_text = " | ".join(
                    f"{name}: {value:.6f}" for name, value in loss_values.items()
                )
                print(f"iteration {iteration} | lr: {lr:.6e} | {loss_text}", flush=True)

    if progress is not None:
        progress.close()

    return iteration


