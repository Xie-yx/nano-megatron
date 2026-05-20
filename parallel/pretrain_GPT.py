from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from parallel.global_vars import get_args
from parallel.model.GPT_model import ParallelGPT
from parallel.training import pretrain


def model_provider(pre_process=True, post_process=True):
    return ParallelGPT(parallel_output=False)


def train_valid_test_dataset_provider():
    return None, None, None


def _build_causal_mask(batch_size, seq_length, device):
    mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
    return mask.view(1, 1, seq_length, seq_length).expand(
        batch_size, 1, seq_length, seq_length
    )


def get_batch(data_iterator):
    args = get_args()
    tokens = next(data_iterator)
    if tokens is None:
        raise RuntimeError("data_iterator returned None.")

    seq_length = args.seq_length
    sample_length = seq_length + 1
    if len(tokens) < sample_length:
        raise ValueError(
            f"Dataset is too short for seq_length={seq_length}: got {len(tokens)} tokens."
        )

    starts = torch.randint(
        0,
        len(tokens) - sample_length + 1,
        (args.micro_batch_size,),
    )
    samples = [
        torch.from_numpy(np.asarray(tokens[start : start + sample_length], dtype=np.int64))
        for start in starts.tolist()
    ]
    samples = torch.stack(samples, dim=0)

    device = torch.device(args.device)
    samples = samples.to(device=device, dtype=torch.long, non_blocking=True)

    input_ids = samples[:, :-1].contiguous()
    labels = samples[:, 1:].contiguous()

    batch_size, cur_seq_length = input_ids.size()
    position_ids = torch.arange(
        cur_seq_length, device=device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, cur_seq_length)
    attention_mask = _build_causal_mask(batch_size, cur_seq_length, device)

    return input_ids, labels, position_ids, attention_mask


def loss_func(labels, output_tensor):
    loss = F.cross_entropy(
        output_tensor.view(-1, output_tensor.size(-1)),
        labels.view(-1),
    )
    return loss, {"lm loss": loss.detach()}


def forward_step(data_iterator, model):
    input_ids, labels, position_ids, attention_mask = get_batch(data_iterator)
    output_tensor = model(
        input_ids,
        position_ids,
        attention_mask,
        forward_method_parallel_output=False,
    )
    return output_tensor, partial(loss_func, labels)


if __name__ == "__main__":
    pretrain(
        model_provider,
        train_valid_test_dataset_provider,
        forward_step,
    )
