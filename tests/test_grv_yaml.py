import importlib



def test_yaml_weights_loaded() -> None:
    # reload module to ensure YAML is read
    mod = importlib.reload(importlib.import_module('ugh3_metrics.metrics.grv_v4'))
    weights = mod.GrvV4.DEFAULT_WEIGHTS
    assert weights == (0.5, 0.3, 0.2)

