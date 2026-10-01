"""Correctness checks against an FP32 reference."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    has_nan_inf: bool
    mismatches: int
    message: str


def check(out, ref, *, rtol: float, atol: float, reject_nan_inf: bool = True) -> CorrectnessResult:
    """Elementwise |out - ref| <= atol + rtol * |ref| (torch.allclose semantics).

    Relative error is reported only over elements with |ref| > atol, where it is meaningful.
    Non-finite outputs at positions where the reference is finite always count as mismatches.
    """
    import torch

    if tuple(out.shape) != tuple(ref.shape):
        return CorrectnessResult(False, float("inf"), float("inf"), False, -1,
                                 f"shape mismatch: {tuple(out.shape)} vs {tuple(ref.shape)}")
    o, r = out.float(), ref.float()
    finite = torch.isfinite(o)
    has_nan_inf = not bool(finite.all())
    diff = (o - r).abs()
    bad = (diff > atol + rtol * r.abs()) | (~finite & torch.isfinite(r))
    mismatches = int(bad.sum())
    max_abs = float(diff[finite].max()) if bool(finite.any()) else float("inf")
    significant = finite & (r.abs() > atol)
    max_rel = float((diff[significant] / r.abs()[significant]).max()) if bool(significant.any()) else 0.0
    passed = mismatches == 0 and not (reject_nan_inf and has_nan_inf)
    if passed:
        message = "ok"
    else:
        parts = []
        if mismatches:
            parts.append(f"{mismatches}/{o.numel()} elements outside tolerance (rtol={rtol}, atol={atol})")
        if has_nan_inf:
            parts.append(f"{int((~finite).sum())} non-finite outputs")
        message = "; ".join(parts) + f"; max_abs_err={max_abs:.4g}"
    return CorrectnessResult(passed, max_abs, max_rel, has_nan_inf, mismatches, message)
