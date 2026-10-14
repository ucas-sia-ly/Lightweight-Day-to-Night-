import numpy as np
import torch

from a2b_lightvpr.references import models


def test_suite_reuses_only_matching_gallery_and_keeps_query_metrics_separate(monkeypatch):
    encoder, transform, cache = object(), object(), {}
    database = [models.SuiteRow("a", 0, 0), models.SuiteRow("b", 100, 0)]
    query_a = [models.SuiteRow("qa", 0, 0)]
    query_b = [models.SuiteRow("qb", 100, 0)]
    calls = []

    def extract(model, rows, *args):
        calls.append(tuple(row.source_path for row in rows))
        return np.eye(2, dtype=np.float32) if len(rows) == 2 else np.asarray([[1, 0]], dtype=np.float32)

    monkeypatch.setattr(models, "extract_descriptors", extract)

    def evaluate(db, queries):
        return models.evaluate_dual_encoder(encoder, encoder, db, queries, transform, 25, None,
                                           (1,), torch.device("cpu"), 2, 0, cache)

    assert evaluate(database, query_a)["recalls"]["1"] == 1
    assert evaluate(list(database), query_b)["recalls"]["1"] == 0
    assert calls == [("a", "b"), ("qa",), ("qb",)]
    other = [models.SuiteRow("c", 0, 0), models.SuiteRow("d", 100, 0)]
    evaluate(other, query_a)
    assert len(cache) == 1
    assert next(iter(cache))[2] == ("c", "d")


def test_large_gallery_is_not_retained_in_suite_cache(monkeypatch):
    class LargeReportedArray(np.ndarray):
        @property
        def nbytes(self):
            return 257 * 1024**2

    gallery = np.eye(2, dtype=np.float32).view(LargeReportedArray)
    database = [models.SuiteRow("a", 0, 0), models.SuiteRow("b", 100, 0)]
    queries = [models.SuiteRow("q", 0, 0)]
    monkeypatch.setattr(models, "extract_descriptors", lambda model, rows, *args:
                        gallery if len(rows) == 2 else np.asarray([[1, 0]], dtype=np.float32))
    cache = {}
    result = models.evaluate_dual_encoder(None, None, database, queries, None, 25, None,
                                         (1,), torch.device("cpu"), 2, 0, cache)
    assert result["recalls"]["1"] == 1 and not cache
