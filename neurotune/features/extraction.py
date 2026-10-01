"""Feature extraction for (workload shape, kernel configuration) pairs.

Features are cheap, analytical descriptions of the work a configuration does:
tile counts, wave quantization, padding waste, estimated memory traffic and
on-chip resource use. They are computed without touching the GPU.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..search.space import DeviceLimits, KernelConfig, Shape, acc_per_thread, shared_mem_bytes

FEATURE_NAMES: tuple[str, ...] = (
    "log2_m", "log2_n", "log2_k",
    "log2_block_m", "log2_block_n", "log2_block_k",
    "group_m_eff", "num_warps", "num_stages",
    "log2_tiles", "waves", "last_wave_util", "pad_efficiency", "k_iters",
    "log2_flops", "log2_traffic", "log2_intensity",
    "smem_fraction", "acc_per_thread",
)

# Simple analytical baseline: operation count, memory traffic and occupancy proxies only.
ANALYTICAL_FEATURES: tuple[str, ...] = ("log2_flops", "log2_traffic", "waves", "pad_efficiency")


def feature_dict(shape: Shape, cfg: KernelConfig, device: DeviceLimits, dtype_bytes: int) -> dict[str, float]:
    m, n, k = shape
    tm, tn, tk = math.ceil(m / cfg.block_m), math.ceil(n / cfg.block_n), math.ceil(k / cfg.block_k)
    tiles = tm * tn
    sms = max(1, device.sm_count)
    waves = tiles / sms
    last_wave_util = tiles / (math.ceil(waves) * sms)
    padded = (tm * cfg.block_m) * (tn * cfg.block_n) * (tk * cfg.block_k)
    flops = 2.0 * m * n * k
    # Without L2 reuse, A is re-read once per column of tiles and B once per row of tiles.
    traffic = dtype_bytes * (m * k * tn + k * n * tm + m * n)
    return {
        "log2_m": math.log2(m), "log2_n": math.log2(n), "log2_k": math.log2(k),
        "log2_block_m": math.log2(cfg.block_m), "log2_block_n": math.log2(cfg.block_n),
        "log2_block_k": math.log2(cfg.block_k),
        "group_m_eff": float(min(cfg.group_m, tm)),
        "num_warps": float(cfg.num_warps), "num_stages": float(cfg.num_stages),
        "log2_tiles": math.log2(tiles), "waves": waves, "last_wave_util": last_wave_util,
        "pad_efficiency": (m * n * k) / padded, "k_iters": float(tk),
        "log2_flops": math.log2(flops), "log2_traffic": math.log2(traffic),
        "log2_intensity": math.log2(flops / traffic),
        "smem_fraction": shared_mem_bytes(cfg, dtype_bytes) / device.max_shared_mem_per_block,
        "acc_per_thread": acc_per_thread(cfg),
    }


def feature_vector(shape: Shape, cfg: KernelConfig, device: DeviceLimits, dtype_bytes: int,
                   names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    d = feature_dict(shape, cfg, device, dtype_bytes)
    return np.array([d[name] for name in names], dtype=np.float64)


def feature_matrix(pairs: Sequence[tuple[Shape, KernelConfig]], device: DeviceLimits, dtype_bytes: int,
                   names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    if not pairs:
        return np.zeros((0, len(names)))
    return np.stack([feature_vector(s, c, device, dtype_bytes, names) for s, c in pairs])
