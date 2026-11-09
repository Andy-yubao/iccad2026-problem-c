"""Capture-safe two-axis product, preserving PyTorch 2.7.1 backward arithmetic.

PyTorch FunctionsManual.cpp prod_backward(dim) calls sum().item() to choose a
zero-safe path. Keep that choice on CUDA rather than synchronizing to the host.
Source: pytorch/pytorch v2.7.1, torch/csrc/autograd/FunctionsManual.cpp.
"""
import torch


class _Product2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        if value.shape[-1] != 2: raise ValueError('Only a final two-axis product is supported')
        result = value.prod(-1)
        ctx.save_for_backward(value, result)
        return result

    @staticmethod
    def backward(ctx, grad):
        value, result = ctx.saved_tensors
        # Original selection is global across this entire reduction input.
        has_zero = (value == 0).any()
        safe = grad[..., None] * torch.stack((value[..., 1], value[..., 0]), -1)
        ordinary = grad[..., None] * (result[..., None] / value)
        return torch.where(has_zero, safe, ordinary)


def product2(value):
    return _Product2.apply(value)
