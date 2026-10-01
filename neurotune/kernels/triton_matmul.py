"""Parameterized Triton matrix multiplication: C[M, N] = A[M, K] @ B[K, N].

* FP16/BF16 inputs, FP32 accumulation, output in the input dtype.
* All loads and stores are masked, so arbitrary (non-aligned) M, N, K are supported.
* Grouped program ordering (GROUP_M) improves L2 reuse of B tiles.

Tunable parameters: BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M (compile-time constants)
and num_warps, num_stages (launch options).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..search.space import KernelConfig


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor, config: KernelConfig, out: torch.Tensor | None = None) -> torch.Tensor:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2-D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)} @ {tuple(b.shape)}")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"expected matching fp16/bf16 inputs, got {a.dtype} and {b.dtype}")
    if not (a.is_cuda and b.is_cuda):
        raise ValueError("inputs must be CUDA tensors")
    M, K = a.shape
    N = b.shape[1]
    if out is None:
        out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    elif tuple(out.shape) != (M, N) or out.dtype != a.dtype:
        raise ValueError("`out` has the wrong shape or dtype")
    grid = (triton.cdiv(M, config.block_m) * triton.cdiv(N, config.block_n),)
    _matmul_kernel[grid](
        a, b, out, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0), out.stride(1),
        BLOCK_M=config.block_m, BLOCK_N=config.block_n, BLOCK_K=config.block_k, GROUP_M=config.group_m,
        num_warps=config.num_warps, num_stages=config.num_stages,
    )
    return out
