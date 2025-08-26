from __future__ import annotations
import pandas as pd
from tools.paper_report import to_question_level, compute_row_stats, compute_q_stats


def test_question_level_collapse():
    df = pd.DataFrame([
        {"question":"Q1","answer":"A","domain":"x","difficulty":"1","por":0.9,"delta_e":0.2,"grv":0.6},
        {"question":"Q1","answer":"B","domain":"x","difficulty":"1","por":0.8,"delta_e":0.1,"grv":0.7},
        {"question":"Q2","answer":"A","domain":"y","difficulty":"2","por":0.7,"delta_e":0.3,"grv":0.5},
    ])
    rows = compute_row_stats(df)
    assert rows["rows"] == 3
    dfq = to_question_level(df)
    qs = compute_q_stats(dfq)
    assert qs["questions"] == 2
    # mean of Q1(A/B) should be ~0.85 for PoR
    assert abs(float(dfq[dfq["question"]=="Q1"]["por"]) - 0.85) < 1e-6
