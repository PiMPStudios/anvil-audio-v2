import os
import random

import numpy as np
import torch

from anvil_audio.utils.memory import flush_memory_caches


def exists(x: torch.Tensor):
    return x is not None


def get_best_device() -> torch.device:
    """Return the best available compute device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def empty_cache(device: torch.device = None):
    """Free unused memory on the given device (or best available)."""
    if device is None:
        device = get_best_device()
    flush_memory_caches(device=device, include_gc=False, include_mlx=False)


def get_world_size():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    else:
        return torch.distributed.get_world_size()


def get_rank():
    """Get rank of current process."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    else:
        return torch.distributed.get_rank()


def print_once(*args):
    if get_rank() == 0:
        print(*args)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters()) + \
        sum(p.numel() for p in model.buffers())


def copy_state_dict(model, state_dict):
    """Load state_dict to model, but only for keys that match exactly.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    for key in state_dict:
        if key in model_state_dict and state_dict[key].shape == model_state_dict[key].shape:
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[key] = state_dict[key]

    model.load_state_dict(model_state_dict, strict=False)
