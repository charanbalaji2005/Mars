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


@pytest.mark.parametrize("shape", SMOKE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_default_config_matches_reference(shape):
    a, b = MatmulWorkload(shape, "fp16", 0).make_inputs("cuda")
    out = matmul(a, b, DEFAULT_CONFIG)
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


def test_check_detects_errors():
    ref = torch.ones(4, 4, device="cuda")
    bad = ref.clone()
    bad[0, 0] = 2.0
    assert not check(bad, ref, rtol=0.01, atol=0.01).passed
    bad[0, 0] = float("nan")
    assert not check(bad, ref, rtol=0.01, atol=0.01).passed
    assert check(ref.half(), ref, rtol=0.01, atol=0.01).passed
