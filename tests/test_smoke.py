import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def test_imports() -> None:
    from ugh3_metrics_lib import facade, core
    _ = facade.trigger
    _ = core.deltae
    _ = core.grv
