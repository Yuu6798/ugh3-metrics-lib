import numpy as np
import pytest

from core.metrics import calc_delta_e_internal


def test_delta_e_internal_identical() -> None:
    v1 = np.array([1.0, 0.0])
    assert calc_delta_e_internal(v1, v1) == 0.0


def test_delta_e_internal_orthogonal() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.0, 1.0])
    assert pytest.approx(calc_delta_e_internal(v1, v2), abs=0.05) == 1.0


def test_delta_e_internal_clip() -> None:
    v1 = np.array([1.0, 0.0])
    v2 = np.array([-1.0, 0.0])
    assert calc_delta_e_internal(v1, v2) == 1.0


def test_delta_e_internal_random_unit_vectors() -> None:
    rng = np.random.default_rng(42)
    v1 = rng.normal(size=768)
    v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=768)
    v2 /= np.linalg.norm(v2)
    de = calc_delta_e_internal(v1, v2)
    assert 0.0 <= de <= 1.0
