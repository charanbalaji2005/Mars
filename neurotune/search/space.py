"""Search space for the parameterized Triton matmul kernel.

The space is a discrete grid. Static constraints remove configurations that are
known to be illegal or pointless *before* any GPU time is spent; every pruned
configuration is counted by reason so reports can show what was excluded.
"""
from __future__ import annotations

import hashlib
import itertools
import random
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import yaml

from ..errors import ConfigError

Shape = tuple[int, int, int]
PARAM_NAMES = ("block_m", "block_n", "block_k", "group_m", "num_warps", "num_stages")
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


@dataclass(frozen=True, order=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int

    def key(self) -> str:
        return (f"bm{self.block_m}_bn{self.block_n}_bk{self.block_k}"
                f"_g{self.group_m}_w{self.num_warps}_s{self.num_stages}")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "KernelConfig":
        return cls(**{name: int(data[name]) for name in PARAM_NAMES})


# A fixed, conventional configuration used as the "default Triton" baseline.
DEFAULT_CONFIG = KernelConfig(block_m=64, block_n=64, block_k=32, group_m=8, num_warps=4, num_stages=3)


@dataclass(frozen=True)
class DeviceLimits:
    name: str
    compute_capability: tuple[int, int]
    sm_count: int
    max_shared_mem_per_block: int
    total_memory_bytes: int
    l2_cache_bytes: int

    def to_dict(self) -> dict:
        data = asdict(self)
        data["compute_capability"] = list(self.compute_capability)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "DeviceLimits":
        return cls(
            name=str(data["name"]),
            compute_capability=tuple(int(x) for x in data["compute_capability"]),  # type: ignore[arg-type]
            sm_count=int(data["sm_count"]),
            max_shared_mem_per_block=int(data["max_shared_mem_per_block"]),
            total_memory_bytes=int(data["total_memory_bytes"]),
            l2_cache_bytes=int(data["l2_cache_bytes"]),
        )

    @classmethod
    def assumed_rtx4050_laptop(cls) -> "DeviceLimits":
        """Nominal RTX 4050 Laptop limits, for offline planning and CPU-only tests.

        Real experiments always use limits discovered from the device.
        """
        return cls("NVIDIA GeForce RTX 4050 Laptop GPU (assumed)", (8, 9), 20,
                   101_376, 6 * 1024**3, 24 * 1024**2)


@dataclass(frozen=True)
class SpaceConstraints:
    # fp32 accumulator registers per thread; above this, spills are near-certain.
    max_acc_regs_per_thread: int = 256
    # Tiles so small that threads hold fewer than this many outputs waste the warps.
    min_acc_per_thread: int = 2
    # Blocks larger than the next power of two of a dimension only add masked work.
    prune_oversized_blocks: bool = True


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


@dataclass
class SearchSpace:
    block_m: tuple[int, ...] = (16, 32, 64, 128, 256)
    block_n: tuple[int, ...] = (16, 32, 64, 128, 256)
    block_k: tuple[int, ...] = (16, 32, 64, 128)
    group_m: tuple[int, ...] = (1, 4, 8)
    num_warps: tuple[int, ...] = (2, 4, 8)
    num_stages: tuple[int, ...] = (2, 3, 4, 5)
    constraints: SpaceConstraints = field(default_factory=SpaceConstraints)

    def __post_init__(self) -> None:
        for name in PARAM_NAMES:
            values = tuple(int(v) for v in getattr(self, name))
            if not values or any(v < 1 for v in values) or len(set(values)) != len(values):
                raise ConfigError(f"search space '{name}' must be a non-empty list of unique positive ints")
            setattr(self, name, tuple(sorted(values)))
        for name in ("block_m", "block_n", "block_k"):
            if any(v < 16 or v & (v - 1) for v in getattr(self, name)):
                raise ConfigError(f"search space '{name}' values must be powers of two >= 16 (tl.dot requirement)")
        if any(w & (w - 1) for w in self.num_warps):
            raise ConfigError("search space 'num_warps' values must be powers of two")

    @property
    def raw_size(self) -> int:
        size = 1
        for name in PARAM_NAMES:
            size *= len(getattr(self, name))
        return size

    def all_configs(self) -> Iterator[KernelConfig]:
        for values in itertools.product(*(getattr(self, n) for n in PARAM_NAMES)):
            yield KernelConfig(*values)

    def contains(self, cfg: KernelConfig) -> bool:
        return all(getattr(cfg, n) in getattr(self, n) for n in PARAM_NAMES)

    def check(self, cfg: KernelConfig, device: DeviceLimits, dtype_bytes: int,
              shape: Shape | None = None) -> str | None:
        """Return None if `cfg` is statically valid, else a short reason string."""
        if not self.contains(cfg):
            return "not_in_space"
        reason = check_hardware(cfg, device, dtype_bytes, self.constraints)
        if reason is None and shape is not None:
            reason = check_shape(cfg, shape, self.constraints)
        return reason

    def valid_configs(self, device: DeviceLimits, dtype_bytes: int,
                      shape: Shape | None = None) -> list[KernelConfig]:
        return sorted(c for c in self.all_configs() if self.check(c, device, dtype_bytes, shape) is None)

    def prune_report(self, device: DeviceLimits, dtype_bytes: int, shape: Shape | None = None) -> dict:
        reasons = Counter(self.check(c, device, dtype_bytes, shape) or "valid" for c in self.all_configs())
        return {"raw_size": self.raw_size, **dict(sorted(reasons.items()))}

    def to_dict(self) -> dict:
        data: dict = {name: list(getattr(self, name)) for name in PARAM_NAMES}
        data["constraints"] = asdict(self.constraints)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SearchSpace":
        data = dict(data or {})
        unknown = set(data) - set(PARAM_NAMES) - {"constraints"}
        if unknown:
            raise ConfigError(f"search space has unknown key(s): {sorted(unknown)}")
        constraints = SpaceConstraints(**(data.pop("constraints", None) or {}))
        return cls(**{k: tuple(v) for k, v in data.items()}, constraints=constraints)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchSpace":
        try:
            return cls.from_dict(yaml.safe_load(Path(path).read_text()))
        except (OSError, yaml.YAMLError, TypeError) as exc:
            raise ConfigError(f"could not load search space from {path}: {exc}") from exc


def shared_mem_bytes(cfg: KernelConfig, dtype_bytes: int) -> int:
    """Estimated shared memory for the pipelined A/B tiles (one buffer per stage)."""
    return (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * dtype_bytes * cfg.num_stages


def acc_per_thread(cfg: KernelConfig) -> float:
    return cfg.block_m * cfg.block_n / (cfg.num_warps * 32)


def check_hardware(cfg: KernelConfig, device: DeviceLimits, dtype_bytes: int,
                   constraints: SpaceConstraints = SpaceConstraints()) -> str | None:
    if shared_mem_bytes(cfg, dtype_bytes) > device.max_shared_mem_per_block:
        return "shared_memory"
    acc = acc_per_thread(cfg)
    if acc > constraints.max_acc_regs_per_thread:
        return "register_pressure"
    if acc < constraints.min_acc_per_thread:
        return "underfilled_tile"
    return None


def check_shape(cfg: KernelConfig, shape: Shape,
                constraints: SpaceConstraints = SpaceConstraints()) -> str | None:
    if not constraints.prune_oversized_blocks:
        return None
    m, n, k = shape
    if (cfg.block_m > max(16, next_pow2(m)) or cfg.block_n > max(16, next_pow2(n))
            or cfg.block_k > max(16, next_pow2(k))):
        return "oversized_block"
    return None


def load_space(space_file: str = "") -> SearchSpace:
    return SearchSpace.from_yaml(space_file) if space_file else SearchSpace()


def derive_seed(*parts) -> int:
    """Stable 32-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    blob = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def seeded_order(candidates: Sequence[KernelConfig], seed: int) -> list[KernelConfig]:
    """Deterministic random permutation. Random and learned search share it, so
    with the same seed both start from an identical initial design."""
    order = sorted(candidates)
    random.Random(seed).shuffle(order)
    return order
