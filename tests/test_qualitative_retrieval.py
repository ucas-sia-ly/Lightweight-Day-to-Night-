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


def test_official_mixvpr_qualitative_uses_explicit_compatible_initialization(tmp_path):
    from dataclasses import replace
    from a2b_lightvpr.config import DataConfig, ExperimentConfig
    from a2b_lightvpr.qualitative.export import _official_mixvpr_checkpoint

    checkpoint = tmp_path / "resnet50_MixVPR_512_channels(256)_rows(2).ckpt"
    checkpoint.write_bytes(b"checkpoint fixture")
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    config = replace(config, vpr=replace(config.vpr, model_family="mixvpr", init_checkpoint=str(checkpoint)))
    assert _official_mixvpr_checkpoint(config) == checkpoint
    incompatible = replace(config, vpr=replace(config.vpr, init_checkpoint=str(tmp_path / "adapted_model.pt")))
    with pytest.raises(ValueError, match="initialization checkpoint"):
        _official_mixvpr_checkpoint(incompatible)


def test_missing_official_mixvpr_qualitative_reports_exact_resource(tmp_path, monkeypatch):
    from a2b_lightvpr.config import DataConfig, ExperimentConfig
    from a2b_lightvpr.qualitative import export

    monkeypatch.setattr(export, "PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="third_party/weights/resnet50_MixVPR_512"):
        export._official_mixvpr_checkpoint(ExperimentConfig(DataConfig("robot.csv", "protocol.json")))


def test_qualitative_preserves_preferred_positive_after_dataset_relocation(tmp_path, monkeypatch):
    import json
    from PIL import Image
    from a2b_lightvpr import paths
    from a2b_lightvpr.qualitative import export
    from a2b_lightvpr.references.models import SuiteRow

    datasets = tmp_path / "datasets"
    source_root = datasets / "svox"
    source_root.mkdir(parents=True)
    files = []
    for name, color in (("closest.png", "red"), ("preferred.png", "blue"), ("query.png", "green")):
        path = source_root / name
        Image.new("RGB", (8, 8), color).save(path)
        files.append(path)
    monkeypatch.setattr(paths, "DATASETS_ROOT", datasets)
    monkeypatch.delenv("A2B_DATASETS_ROOT", raising=False)
    database = [SuiteRow(str(files[0]), 0.0, 0.0), SuiteRow(str(files[1]), 5.0, 0.0)]
    queries = [SuiteRow(str(files[2]), 0.0, 0.0)]
    monkeypatch.setattr(export, "suite_rows", lambda *args: (database, queries))
    monkeypatch.setattr(export, "_models", lambda *args: iter(()))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"examples": [{
        "dataset": "svox_night", "query_index": 0,
        "positive_reference_path": str(paths.LEGACY_DATASETS_ROOT / "svox/preferred.png"),
    }]}), encoding="utf-8")
    before = selection.read_bytes()
    output = export.export_qualitative_top1(None, selection, tmp_path / "output", tmp_path / "cache", models=())
    result = json.loads(output.read_text(encoding="utf-8"))["examples"]["svox_night"]
    assert result["ground_truth_positive_index"] == 1
    assert result["ground_truth_positive_path"] == str(files[1])
    assert selection.read_bytes() == before


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


def test_search_exports_three_labelled_rows_nine_real_top1s_and_reuses_cache(tmp_path, monkeypatch):
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
    assert len(results['examples']) == 3
    assert 'robotcar_seasons_night' in results['excluded']
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


def test_public_seasons_qualitative_exports_rankings_without_correctness(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from PIL import Image
    from a2b_lightvpr.qualitative import export
    from a2b_lightvpr.references.models import FixedResizeTransform

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    rows = []
    for index, color in enumerate(("red", "green", "blue")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (16, 16), color).save(path)
        rows.append(SimpleNamespace(source_path=str(path), image_id=path.name,
                                   easting=None, northing=None, yaw_deg=None,
                                   ground_truth_available=False))
    database, queries = rows[:2], rows[2:]
    monkeypatch.setattr(export, "suite_rows", lambda *args: (database, queries))
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"synthetic fixture")
    model = torch.nn.Module()
    model.descriptor_dim, model.qualitative_batch_size = 2, None
    model.qualitative_artifacts = {}
    monkeypatch.setattr(export, "_models", lambda *args: iter([
        ("cdnd", model, FixedResizeTransform(16, 16), checkpoint)]))
    monkeypatch.setattr(export, "_cache_descriptors", lambda *args: np.eye(2, dtype=np.float32))
    monkeypatch.setattr(export, "extract_descriptors", lambda *args: np.asarray([[0, 1]], dtype=np.float32))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"examples": [{"dataset": "robotcar_seasons_night", "query_index": 0}]}))
    output = tmp_path / "output"
    result = json.loads(export.export_qualitative_top1(
        None, selection, output, tmp_path / "cache", models=("cdnd",)).read_text())
    example = result["examples"]["robotcar_seasons_night"]
    assert example["ground_truth_available"] is False
    assert example["evaluation_status"] == "retrieval_only_no_public_test_gt"
    assert "ground_truth_positive_index" not in example
    prediction = example["models"]["cdnd"]
    assert prediction["success_at_1_5_10"] is None
    assert prediction["top1_distance_m"] is None
    assert [entry["database_index"] for entry in prediction["top_k"]] == [1, 0]
    assert (output / "robotcar_seasons_night/comparison.jpg").is_file()
    assert not list(output.rglob("ground_truth_positive.*"))
    with pytest.raises(ValueError, match="ground truth is unavailable"):
        export._positives(database, queries[0], export.DATASETS["robotcar_seasons_night"])
