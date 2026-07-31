import importlib.util

import pytest

from a2b_lightvpr.runtime import _require_onnx_dependencies


def test_onnx_export_dependency_error_is_actionable(monkeypatch):
    available = {"onnx": True, "onnxscript": False, "onnxruntime": False}
    original = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in available:
            return object() if available[name] else None
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ModuleNotFoundError, match=r"onnxscript, onnxruntime.*\[export\]"):
        _require_onnx_dependencies()
