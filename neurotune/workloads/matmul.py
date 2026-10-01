"""Matmul workload definitions and test suites."""
from __future__ import annotations

from dataclasses import dataclass

from ..search.space import DTYPE_BYTES, Shape, derive_seed

# Includes non-aligned and degenerate dimensions on purpose.
SMOKE_SHAPES: tuple[Shape, ...] = (
    (16, 16, 16), (64, 64, 64), (128, 128, 128), (257, 511, 129),
    (1, 1024, 7), (1000, 33, 517), (512, 512, 512),
)
FULL_SHAPES: tuple[Shape, ...] = SMOKE_SHAPES + (
    (1024, 1024, 1024), (2048, 512, 1024), (4096, 4096, 64), (31, 4097, 255),
    (768, 3072, 768), (2048, 2048, 2048),
)
SUITES = {"smoke": SMOKE_SHAPES, "full": FULL_SHAPES}
_TORCH_DTYPES = {"fp16": "float16", "bf16": "bfloat16"}


@dataclass(frozen=True)
class MatmulWorkload:
    shape: Shape
    dtype: str = "fp16"
    seed: int = 0

    @property
    def flops(self) -> float:
        m, n, k = self.shape
        return 2.0 * m * n * k

    @property
    def name(self) -> str:
        return f"matmul_{self.dtype}_" + "x".join(map(str, self.shape))

    def memory_estimate_bytes(self) -> int:
        """Inputs + output + Triton out buffer + FP32 reference + torch.matmul output."""
        m, n, k = self.shape
        e = DTYPE_BYTES[self.dtype]
        return (m * k + k * n) * e + 3 * m * n * e + m * n * 4

    def make_inputs(self, device: str = "cuda"):
        """Deterministic N(0, 1) inputs generated on CPU (device-independent values)."""
        import torch

        dtype = getattr(torch, _TORCH_DTYPES[self.dtype])
        gen = torch.Generator(device="cpu").manual_seed(derive_seed(self.seed, *self.shape))
        m, n, k = self.shape
        a = torch.randn((m, k), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        b = torch.randn((k, n), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
        return a.contiguous(), b.contiguous()
