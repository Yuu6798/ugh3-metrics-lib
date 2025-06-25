import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_imports() -> None:
    import facade.trigger
    import core.deltae
    import core.grv
