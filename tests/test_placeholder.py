import pytest


@pytest.mark.xfail(reason="このテストが残っている限り CI は通りません", strict=True)
def test_remove_me() -> None:
    assert False
