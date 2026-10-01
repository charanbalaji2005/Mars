import pytest

from neurotune.benchmark.runner import classify_exception


class OutOfResources(Exception):
    pass


class CompilationError(Exception):
    pass


@pytest.mark.parametrize("exc,expected", [
    (OutOfResources("out of resource: shared memory, Required: 120000"), "out_of_resources"),
    (RuntimeError("CUDA error: an illegal memory access was encountered"), "fatal"),
    (MemoryError("out of memory: shape needs 9 GiB"), "oom"),
    (CompilationError("at 12:4: invalid"), "compile_error"),
    (ValueError("something else"), "runtime_error"),
])
def test_classify(exc, expected):
    assert classify_exception(exc) == expected
