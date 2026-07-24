import numpy as np
import pytest

from a2b_lightvpr.generation_evaluation import bootstrap_summary, polynomial_mmd


def test_bootstrap_summary_is_deterministic_and_contains_mean():
    first = bootstrap_summary([1.0, 2.0, 3.0], samples=100, seed=4)
    second = bootstrap_summary([1.0, 2.0, 3.0], samples=100, seed=4)
    assert first == second
    assert first["mean"] == pytest.approx(2.0)
    assert first["count"] == 3


def test_polynomial_mmd_is_near_zero_for_same_features():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 8))
    assert polynomial_mmd(features, features) == pytest.approx(0.0, abs=0.1)
