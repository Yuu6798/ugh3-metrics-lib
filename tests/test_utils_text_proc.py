from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ugh3_metrics.utils import tokenize, entropy, pmi


def test_tokenize() -> None:
    assert tokenize("a b c") == ["a", "b", "c"]
    assert tokenize(["a b", "c"]) == ["a", "b", "c"]


def test_entropy_smoke() -> None:
    tokens = "alpha beta beta gamma".split()
    val = entropy(tokens)
    assert 0.0 <= val <= 1.0


def test_pmi_smoke() -> None:
    tokens = "alpha beta beta gamma".split()
    val = pmi(tokens, window=2)
    assert isinstance(val, float)
