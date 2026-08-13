import numpy as np

from a2b_lightvpr.evaluation import retrieval_metrics


def test_retrieval_metrics_known_ranking():
    database = np.eye(3, dtype=np.float32)
    queries = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    metrics = retrieval_metrics(database, queries, [np.asarray([0]), np.asarray([1])], (1, 5))
    assert metrics["recalls"]["1"] == 1.0
    assert metrics["recalls"]["5"] == 1.0
    assert metrics["valid_by_query"] == [1, 1]


def test_invalid_queries_are_explicit_and_excluded_from_recall():
    database = np.eye(2, dtype=np.float32)
    queries = np.eye(2, dtype=np.float32)
    metrics = retrieval_metrics(
        database, queries, [np.asarray([0]), np.asarray([])], (1,)
    )
    assert metrics["valid_by_query"] == [1, 0]
    assert metrics["recalls"]["1"] == 1.0
