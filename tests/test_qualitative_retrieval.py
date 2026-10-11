import numpy as np
import pytest

from a2b_lightvpr.cli import _parser
from a2b_lightvpr.qualitative.export import _top_indices


def test_top_indices_returns_descending_inner_product_order():
    database = np.asarray([[1, 0], [0, 1], [.8, .2]], dtype=np.float32)
    indices, scores = _top_indices(database, np.asarray([1, 0], dtype=np.float32), 3)
    assert indices.tolist() == [0, 2, 1]
    assert scores.tolist() == pytest.approx([1.0, .8, 0.0])


def test_qualitative_cli_contract():
    args = _parser().parse_args([
        "export-qualitative-top1", "--config", "config.yaml",
        "--selection", "selection.json", "--output-dir", "output",
        "--cache-dir", "cache", "--batch-size", "8", "--workers", "0",
    ])
    assert args.command == "export-qualitative-top1"
    assert args.batch_size == 8
    assert args.workers == 0


def test_selection_prefers_all_failures_then_boq_salad_tie():
    from a2b_lightvpr.qualitative.search import rank_candidates
    correct = {
        'cdnd': np.array([False, True, True, True, True]),
        'cosplace': np.array([False, False, True, False, False]),
        'boq_pca': np.array([False, True, False, False, False]),
        'salad': np.array([False, False, False, False, False]),
    }
    # Index 0 is not a CDND hit; index 4 has no valid positive.
    assert rank_candidates(correct, np.array([True, True, True, True, False])) == [3, 1, 2]


def test_search_exports_four_rows_nine_real_top1s_and_reuses_cache(tmp_path, monkeypatch):
    import json
    import torch
    from PIL import Image
    from a2b_lightvpr.qualitative import search as search
    from a2b_lightvpr.references.models import SuiteRow, FixedResizeTransform

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    paths = []
    for i, color in enumerate(('red', 'green', 'blue', 'yellow')):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (32, 24), color).save(path)
        paths.append(str(path))
    db = [SuiteRow(paths[0], 0, 0), SuiteRow(paths[1], 100, 0)]
    queries = [SuiteRow(paths[2], 100, 0), SuiteRow(paths[3], 0, 0)]
    monkeypatch.setattr(search, 'suite_rows', lambda *args: (db, queries))
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'synthetic test fixture')

    def models(config, order):
        for name in order:
            model = torch.nn.Module()
            model.name = name
            model.descriptor_dim = 2
            model.qualitative_batch_size = 1 if name == 'crica_pca' else None
            model.qualitative_artifacts = {}
            model.qualitative_spec = {}
            yield name, model, FixedResizeTransform(32, 32), checkpoint

    def cache(model, image_rows, transform, device, batch, workers, path, identity):
        if model.name == 'crica_pca':
            assert batch == 1
        if identity['role'] == 'database':
            return np.eye(2, dtype=np.float32)
        return np.array([[1, 0], [1, 0]] if model.name == 'cdnd' else [[0, 1], [0, 1]], dtype=np.float32)

    monkeypatch.setattr(search, '_models', models)
    monkeypatch.setattr(search, '_cache_descriptors', cache)
    output = tmp_path / 'output'
    selection = tmp_path / 'selection.json'
    result_path = search.search_qualitative_top1(None, selection, output, tmp_path / 'cache')
    results = json.loads(result_path.read_text())
    assert len(results['examples']) == 4
    for example in results['examples'].values():
        assert example['query_index'] == 1
        assert len(example['models']) == 9
        assert example['models']['cdnd']['correct_at_1']
        assert example['models']['cdnd']['top1_path'] == paths[0]
        assert all(not v['correct_at_1'] for k, v in example['models'].items() if k != 'cdnd')
    assert (output / 'comparison_all.png').is_file()
    assert (output / 'comparison_all.pdf').is_file()
    monkeypatch.setattr(search, '_cache_descriptors', lambda *a, **kw: pytest.fail('cache was not reused'))
    search.search_qualitative_top1(None, selection, output, tmp_path / 'cache')
