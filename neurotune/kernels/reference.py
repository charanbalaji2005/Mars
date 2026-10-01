"""PyTorch reference implementations."""
from __future__ import annotations


def reference_fp32(a, b):
    """FP32 matmul with TF32 explicitly disabled, used as the correctness oracle."""
    import torch

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        return torch.matmul(a.float(), b.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
