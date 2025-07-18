import pytest

from core.metrics import POR_FIRE_THRESHOLD, is_por_fire


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.81, False),
        (POR_FIRE_THRESHOLD, True),
        (0.95, True),
    ],
)  # type: ignore[misc]
def test_is_por_fire(score: float, expected: bool) -> None:
    assert is_por_fire(score) is expected
