from dataclasses import replace

import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.vpr.trainer import train_vpr


@pytest.mark.parametrize('artifact', ['model_best.pt', 'checkpoints/latest.pt',
                                     'training_metrics.jsonl', 'validation/latest.json'])
def test_existing_run_is_rejected_before_rewriting_or_loading_models(tmp_path, artifact):
    existing = tmp_path / artifact
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_bytes(b'original experiment evidence')
    config = ExperimentConfig(DataConfig())
    config = replace(config, vpr=replace(config.vpr, output_dir=str(tmp_path), require_cuda=True))
    with pytest.raises(FileExistsError, match='vpr.resume'):
        train_vpr(config)
    assert existing.read_bytes() == b'original experiment evidence'
    assert not (tmp_path / 'resolved_config.json').exists()
