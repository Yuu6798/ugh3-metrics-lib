from facade import trigger
from core import deltae, grv


def test_imports() -> None:
    """Ensure top-level modules are importable."""
    _ = trigger
    _ = deltae
    _ = grv
