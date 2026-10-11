import json

from a2b_lightvpr.references.summary import DAY_MACRO, NIGHT_MACRO, summarize_small_references


def test_small_summary_uses_machine_readable_accuracy_and_resources(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "reference_models:\n"
        "  model:\n"
        "    loader: fixture\n"
        "    backbone: DINOv2-S/14-Reg4\n"
        "    descriptor_dim: 512\n"
        "    official_checkpoint: false\n"
        "    training_status: re-fitted\n",
        encoding="utf-8",
    )
    datasets = {}
    for index, protocol in enumerate((*NIGHT_MACRO, *DAY_MACRO)):
        datasets[protocol] = {
            "recalls": {"1": 0.5 + index / 100, "5": 0.7, "10": 0.8},
            "valid_queries": 10,
            "evaluation_fingerprint": protocol,
        }
    summary = tmp_path / "reference/model/final_suite/native/adapted/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"datasets": datasets}), encoding="utf-8")
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"models": {"model": {"parameters": 10}}}), encoding="utf-8")
    output = tmp_path / "comparison.json"
    summarize_small_references(
        registry, tmp_path / "reference", [resources], output,
        model_names=("model",),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "dinov2_small_comparison_v1"
    assert payload["models"]["model"]["official_checkpoint"] is False
    assert payload["models"]["model"]["resources"]["parameters"] == 10
