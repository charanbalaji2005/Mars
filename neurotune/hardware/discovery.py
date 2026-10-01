"""Hardware/software discovery and the `neurotune doctor` checks."""
from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from ..errors import EnvironmentCheckError
from ..search.space import DeviceLimits

# Max opt-in shared memory per block (bytes) by compute capability, used only as a fallback.
_SMEM_BY_CC = {(7, 0): 98_304, (7, 5): 65_536, (8, 0): 166_912, (8, 6): 101_376,
               (8, 7): 166_912, (8, 9): 101_376, (9, 0): 232_448}
_SMI_FIELDS = ("name", "driver_version", "pstate", "temperature.gpu", "clocks.sm", "clocks.mem",
               "clocks.max.sm", "power.draw", "power.limit", "utilization.gpu", "memory.used", "memory.total")


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "warn" | "fail"
    detail: str


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception:
        return None


def software_versions() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "neurotune": _version("neurotune"), "numpy": _version("numpy"),
        "scikit_learn": _version("sklearn"), "optuna": _version("optuna"),
        "torch": _version("torch"), "triton": _version("triton"),
    }
    try:
        import torch

        info["torch_cuda_runtime"] = torch.version.cuda
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                            timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        info["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        info["git_commit"] = None
    return info


def nvidia_smi_snapshot() -> list[dict[str, str]] | None:
    """Clocks, temperature, power state etc. at this moment (None if nvidia-smi is unavailable)."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except Exception:
        return None
    rows = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(_SMI_FIELDS, values)))
    return rows


def gpu_telemetry_snapshot(device_index: int = 0) -> dict[str, float] | None:
    """Parsed numerical GPU telemetry: temperature (C), clocks (MHz), power (W), utilization (%)."""
    rows = nvidia_smi_snapshot()
    if not rows or device_index >= len(rows):
        return None
    r = rows[device_index]

    def _float(key):
        try:
            val = r.get(key, "").replace("[Not Supported]", "").strip()
            return float(val) if val else float("nan")
        except (ValueError, TypeError):
            return float("nan")

    return {
        "temperature_c": _float("temperature.gpu"),
        "power_w": _float("power.draw"),
        "power_limit_w": _float("power.limit"),
        "clock_sm_mhz": _float("clocks.sm"),
        "clock_mem_mhz": _float("clocks.mem"),
        "clock_max_sm_mhz": _float("clocks.max.sm"),
        "utilization_pct": _float("utilization.gpu"),
    }


def device_limits(device_index: int = 0) -> DeviceLimits:
    import torch

    if not torch.cuda.is_available():
        raise EnvironmentCheckError("CUDA is not available to PyTorch")
    props = torch.cuda.get_device_properties(device_index)
    cc = (int(props.major), int(props.minor))
    smem = None
    try:
        import triton

        smem = int(triton.runtime.driver.active.utils.get_device_properties(device_index)["max_shared_mem"])
    except Exception:
        smem = getattr(props, "shared_memory_per_block_optin", None) or _SMEM_BY_CC.get(cc, 49_152)
    l2 = int(getattr(props, "L2_cache_size", 0) or 50 * 1024**2)
    return DeviceLimits(name=props.name, compute_capability=cc, sm_count=int(props.multi_processor_count),
                        max_shared_mem_per_block=int(smem), total_memory_bytes=int(props.total_memory),
                        l2_cache_bytes=l2)


def hardware_metadata(limits: DeviceLimits, full: bool = True) -> dict[str, Any]:
    meta: dict[str, Any] = {"limits": limits.to_dict()}
    if full:
        meta["nvidia_smi_at_start"] = nvidia_smi_snapshot()
        meta["cpu"] = platform.processor() or platform.machine()
    return meta


def require_gpu(dtype: str = "fp16") -> DeviceLimits:
    """Raise EnvironmentCheckError with an actionable message if the GPU stack is unusable."""
    try:
        import torch
    except ImportError as exc:
        raise EnvironmentCheckError("PyTorch is not installed. Install a CUDA build from https://pytorch.org") from exc
    if not torch.cuda.is_available():
        raise EnvironmentCheckError("PyTorch cannot see a CUDA device (CPU-only build, driver problem, "
                                    "or no GPU). Run `neurotune doctor` for details.")
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise EnvironmentCheckError("Triton is not installed (on Linux it ships with the PyTorch CUDA wheel)") from exc
    limits = device_limits()
    if limits.compute_capability < (7, 0):
        raise EnvironmentCheckError(f"{limits.name} has compute capability {limits.compute_capability}; "
                                    "Triton tensor-core matmul needs >= 7.0")
    if dtype == "bf16" and limits.compute_capability < (8, 0):
        raise EnvironmentCheckError("bf16 requires compute capability >= 8.0")
    return limits


def doctor() -> list[Check]:
    checks: list[Check] = []
    py_ok = sys.version_info >= (3, 10)
    checks.append(Check("python", "ok" if py_ok else "fail", sys.version.split()[0]))
    for module in ("numpy", "sklearn", "optuna", "yaml"):
        v = _version(module)
        checks.append(Check(module, "ok" if v else "fail", v or "not installed"))

    try:
        import torch
    except Exception as exc:
        checks.append(Check("torch", "fail", f"not importable: {exc}"))
        return checks
    checks.append(Check("torch", "ok", f"{torch.__version__} (CUDA runtime {torch.version.cuda})"))
    if torch.version.cuda is None:
        checks.append(Check("cuda", "fail", "this PyTorch build has no CUDA support"))
        return checks
    if not torch.cuda.is_available():
        checks.append(Check("cuda", "fail", "torch.cuda.is_available() is False (driver missing or GPU not visible)"))
        return checks
    checks.append(Check("cuda", "ok", f"{torch.cuda.device_count()} device(s) visible"))

    triton_v = _version("triton")
    checks.append(Check("triton", "ok" if triton_v else "fail", triton_v or "not installed"))
    if platform.system() == "Windows":
        checks.append(Check("platform", "warn", "Triton does not officially support native Windows; use WSL2 or Linux"))

    try:
        limits = device_limits()
        cc_ok = limits.compute_capability >= (7, 0)
        checks.append(Check("device", "ok" if cc_ok else "fail",
                            f"{limits.name}, cc {limits.compute_capability[0]}.{limits.compute_capability[1]}, "
                            f"{limits.sm_count} SMs, {limits.total_memory_bytes / 2**30:.1f} GiB, "
                            f"smem/block {limits.max_shared_mem_per_block // 1024} KiB, "
                            f"L2 {limits.l2_cache_bytes // 2**20} MiB"))
    except Exception as exc:
        checks.append(Check("device", "fail", str(exc)))
        return checks

    smi = nvidia_smi_snapshot()
    checks.append(Check("nvidia-smi", "ok" if smi else "warn",
                        f"driver {smi[0].get('driver_version')}, pstate {smi[0].get('pstate')}" if smi
                        else "unavailable: clocks/temperature will not be recorded"))

    if triton_v:
        try:
            from ..benchmark.correctness import check
            from ..kernels.reference import reference_fp32
            from ..kernels.triton_matmul import matmul
            from ..search.space import DEFAULT_CONFIG
            from ..workloads.matmul import MatmulWorkload

            a, b = MatmulWorkload((257, 129, 65), "fp16", 0).make_inputs("cuda")
            out = matmul(a, b, DEFAULT_CONFIG)
            torch.cuda.synchronize()
            result = check(out, reference_fp32(a, b), rtol=0.01, atol=0.01)
            checks.append(Check("triton_kernel", "ok" if result.passed else "fail",
                                "compiled and matched reference on 257x129x65" if result.passed else result.message))
        except Exception as exc:
            checks.append(Check("triton_kernel", "fail", f"{type(exc).__name__}: {str(exc)[:300]}"))
    return checks
