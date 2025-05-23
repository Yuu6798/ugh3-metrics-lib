from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports() -> None:
    import facade.trigger as _ft
    import core.deltae as _de
    import core.grv as _grv

    assert _ft
    assert _de
    assert _grv
