import numpy as np

from a2b_lightvpr.evaluation import paired_bootstrap, retrieval_metrics


def test_retrieval_metrics_known_ranking():
    database = np.eye(3, dtype=np.float32)
    queries = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    metrics = retrieval_metrics(database, queries, [np.asarray([0]), np.asarray([1])], (1, 5))
    assert metrics["recalls"]["1"] == 1.0
    assert metrics["recalls"]["5"] == 1.0


def test_paired_bootstrap_detects_uniform_improvement():
    result = paired_bootstrap([1] * 100, [0] * 100, samples=500, seed=1)
    assert result["significant"] is True
    assert result["ci95"][0] > 0
