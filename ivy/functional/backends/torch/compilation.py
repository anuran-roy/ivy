"""Collection of PyTorch compilation functions"""

# global
import torch


def compile(
    fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None
):
    return torch.jit.script(fn) if dynamic else torch.jit.trace(fn, example_inputs)
