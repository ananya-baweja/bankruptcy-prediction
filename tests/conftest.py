import copy

import pytest

from bpp.config import get_paths, load_config


@pytest.fixture(scope="session")
def base_cfg():
    return load_config()


@pytest.fixture
def cfg(base_cfg):
    return copy.deepcopy(base_cfg)


@pytest.fixture
def paths(cfg, tmp_path):
    return get_paths(cfg, tmp_path / "data")
