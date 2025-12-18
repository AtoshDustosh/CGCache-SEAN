import torch
import numpy as np
import random

import wandb
import warnings
import functools
from torch.profiler import (
    ProfilerActivity,
    tensorboard_trace_handler,
    schedule,
)
from pathlib import Path


def deprecated(reason="This function is deprecated."):
    """
    A customized decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapped

    return decorator


class DummyProfiler:
    def step(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# ---------------------------- Global Objects ----------------------------
ifprofile: bool
run: wandb.Run


def _setup_reproducibility(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pass


def _setup_wandb(args):
    global run
    run = wandb.init(
        group="global",
    )
    run.config.update(args)


profiler_schedule = schedule(
    wait=100,
    warmup=10,
    active=1,
    repeat=10,
    skip_first=20,
    skip_first_wait=1,
)
profile_dir = Path()


def _setup_profiler(args):
    global ifprofile, profiler_ctx, profiler_schedule, profile_dir
    ifprofile = args.profile

    profile_dir = Path(args.profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)


def setup(args):
    _setup_reproducibility(args)
    _setup_wandb(args)
    _setup_profiler(args)


cache_hits = []
