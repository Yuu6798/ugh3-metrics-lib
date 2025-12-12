from __future__ import annotations

from pathlib import Path
import pytest

# mypy: disable-error-code=unused-ignore


DATA_PATH = Path(__file__).resolve().parents[1] / "datasets" / "current_recalc.parquet"


@pytest.mark.skipif(  # type: ignore[misc,untyped-decorator]  # pytest decorator lacks typing
    not DATA_PATH.exists(),
    reason="datasets/current_recalc.parquet not available",
)
def test_dataset_consistency() -> None:
    """Basic sanity checks for the current dataset."""
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(DATA_PATH, columns=["delta_e_internal", "por_fire"])
    if len(df) == 0:
        pytest.skip("dataset is empty")

    sample_n = min(max(int(len(df) * 0.1), 1), 1000, len(df))
    df = df.sample(n=sample_n, random_state=0)

    assert df["delta_e_internal"].between(0, 1, inclusive="both").all()

    rate = df["por_fire"].mean()
    assert 0.1 <= rate <= 0.4, f"PoR firing rate {rate} outside [0.1, 0.4]"
