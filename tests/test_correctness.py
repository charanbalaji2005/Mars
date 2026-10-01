"""GPU correctness tests for the Triton kernel. Skipped automatically without CUDA + Triton."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from neurotune.benchmark.correctness import check  # noqa: E402
from neurotune.hardware.discovery import device_limits  # noqa: E402
from neurotune.kernels.reference import reference_fp32  # noqa: E402
from neurotune.kernels.triton_matmul import matmul  # noqa: E402
from neurotune.search.space import DEFAULT_CONFIG, SearchSpace, derive_seed, seeded_order  # noqa: E402
from neurotune.workloads.matmul import SMOKE_SHAPES, MatmulWorkload  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("shape", [(16, 16, 16), (128, 128, 128), (257, 511, 129), (1, 1024, 7), (15, 15, 15)],
                         ids=lambda s: "x".join(map(str, s)) if isinstance(s, tuple) else str(s))
def test_default_config_matches_reference(dtype, shape):
    limits = device_limits()
    if dtype == "bf16" and limits.compute_capability < (8, 0):
        pytest.skip("bf16 requires compute capability >= 8.0")
    a, b = MatmulWorkload(shape, dtype, 0).make_inputs("cuda")
    out = matmul(a, b, DEFAULT_CONFIG)
    rtol = 0.02 if dtype == "bf16" else 0.01
    atol = 0.02 if dtype == "bf16" else 0.01
    result = check(out, reference_fp32(a, b), rtol=rtol, atol=atol)
    assert result.passed, result.message


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 16, 1), (37, 73, 101), (1000, 33, 517)],
                         ids=lambda s: "x".join(map(str, s)))
def test_irregular_and_small_shapes(shape):
    a, b = MatmulWorkload(shape, "fp16", 0).make_inputs("cuda")
    out = torch.full((shape[0], shape[1]), float("nan"), dtype=torch.float16, device="cuda")
    matmul(a, b, DEFAULT_CONFIG, out=out)
    result = check(out, reference_fp32(a, b), rtol=0.01, atol=0.01)
    assert result.passed, result.message


@pytest.mark.parametrize("shape", [(257, 511, 129), (1000, 33, 517)], ids=lambda s: "x".join(map(str, s)))
def test_sampled_configs_match_reference(shape):
    limits = device_limits()
    cands = SearchSpace().valid_configs(limits, 2, shape)
    a, b = MatmulWorkload(shape, "fp16", 0).make_inputs("cuda")
    ref = reference_fp32(a, b)
    for cfg in seeded_order(cands, derive_seed("test", *shape))[:8]:
        out = torch.full((shape[0], shape[1]), float("nan"), dtype=torch.float16, device="cuda")
        matmul(a, b, cfg, out=out)
        result = check(out, ref, rtol=0.01, atol=0.01)
        assert result.passed, f"{cfg.key()}: {result.message}"


def test_invalid_config_resource_exhaustion():
    from neurotune.benchmark.runner import classify_exception
    from neurotune.search.space import KernelConfig

    # A deliberately oversized configuration exceeding hardware shared memory / registers
    oversized = KernelConfig(block_m=256, block_n=256, block_k=128, group_m=8, num_warps=8, num_stages=5)
    shape = (256, 256, 256)
    a, b = MatmulWorkload(shape, "fp16", 0).make_inputs("cuda")
    out = torch.empty((shape[0], shape[1]), dtype=torch.float16, device="cuda")
    try:
        matmul(a, b, oversized, out=out)
    except Exception as exc:
        status = classify_exception(exc)
        assert status in ("out_of_resources", "compile_error", "runtime_error")


def test_check_detects_errors():
    ref = torch.ones(4, 4, device="cuda")
    bad = ref.clone()
    bad[0, 0] = 2.0
    assert not check(bad, ref, rtol=0.01, atol=0.01).passed
    bad[0, 0] = float("nan")
    res = check(bad, ref, rtol=0.01, atol=0.01)
    assert not res.passed
    assert res.has_nan_inf
    assert res.mismatches >= 1
    assert check(ref.half(), ref, rtol=0.01, atol=0.01).passed
