import pytest

from neurotune.errors import ConfigError
from neurotune.search.space import (DEFAULT_CONFIG, KernelConfig, SearchSpace, check_hardware, derive_seed,
                                    load_space, seeded_order, shared_mem_bytes)


def test_raw_size_and_validity(limits):
    space = SearchSpace()
    assert space.raw_size == 5 * 5 * 4 * 3 * 3 * 4
    valid = space.valid_configs(limits, 2)
    assert 0 < len(valid) < space.raw_size
    assert all(shared_mem_bytes(c, 2) <= limits.max_shared_mem_per_block for c in valid)
    assert len({c.key() for c in valid}) == len(valid)


def test_default_config_is_hardware_valid(limits):
    assert check_hardware(DEFAULT_CONFIG, limits, 2) is None
    assert SearchSpace().contains(DEFAULT_CONFIG)


def test_shared_memory_limit(limits):
    big = KernelConfig(256, 256, 128, 8, 8, 5)
    assert SearchSpace().check(big, limits, 2) == "shared_memory"


def test_oversized_blocks_pruned_for_small_shapes(limits):
    space = SearchSpace()
    cfg = KernelConfig(256, 64, 32, 8, 4, 3)
    assert space.check(cfg, limits, 2, (128, 128, 128)) == "oversized_block"
    assert space.check(cfg, limits, 2, (1024, 1024, 1024)) is None
    small = space.valid_configs(limits, 2, (16, 16, 16))
    assert small and all(c.block_m == c.block_n == c.block_k == 16 for c in small)


def test_prune_report_accounts_for_everything(limits):
    space = SearchSpace()
    report = space.prune_report(limits, 2, (512, 512, 512))
    assert sum(v for k, v in report.items() if k != "raw_size") == space.raw_size


def test_key_roundtrip():
    cfg = KernelConfig(32, 64, 16, 4, 2, 4)
    assert KernelConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.key() == "bm32_bn64_bk16_g4_w2_s4"


def test_seeded_order_deterministic(limits):
    c = SearchSpace().valid_configs(limits, 2)
    assert seeded_order(c, 3) == seeded_order(list(reversed(c)), 3)
    assert seeded_order(c, 3) != seeded_order(c, 4)
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3) != derive_seed(1, 2, 4)


def test_space_yaml_matches_default():
    assert load_space("configs/search_space.yaml").to_dict() == SearchSpace().to_dict()


def test_invalid_space_rejected():
    with pytest.raises(ConfigError):
        SearchSpace(block_m=(24,))
    with pytest.raises(ConfigError):
        SearchSpace.from_dict({"block_q": [1]})
