#!/usr/bin/env python3
"""PoR・ΔE・grv統合評価型Q&Aロガー.

This module runs a simple Q&A cycle where a user supplied question is
answered by a dummy AI function.  Each turn calculates three metrics:
PoR (Proof of Resonance), ΔE (difference between consecutive answers)
and grv (vocabulary gravity).  The metrics are combined to decide
whether a record should be stored in history.  At the end of the run
all adopted records are written to a CSV file.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from facade.trigger import por_trigger
from core.grv import grv_score as _grv_score

# ---------------------------------------------------------------------------
# Scoring weights and thresholds
# ---------------------------------------------------------------------------
# weight for PoR component in overall score
W_POR: float = 0.4
# weight for (1 - ΔE) component in overall score
W_DE: float = 0.4
# weight for grv component in overall score
W_GRV: float = 0.2
# threshold for adopting a record into history
ADOPT_TH: float = 0.45


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def get_ai_response(question: str) -> str:
    """Return a deterministic AI response for the given question."""
    return f"Answer for '{question}'"


def _similarity(text1: str, text2: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def estimate_ugh_params(question: str, history: List["QaRecord"]) -> Dict[str, Any]:
    """Return automatic UGHer parameters based on the question and history."""
    q_len = len(question)
    h_len = len(history)
    q = min(1.0, q_len / 50.0)
    s = min(1.0, 0.5 + 0.05 * h_len)
    t = 0.5 + min(0.5, q_len / 100.0)
    phi_C = 0.8
    D = min(0.5, 0.1 + 0.02 * h_len)
    return {"q": q, "s": s, "t": t, "phi_C": phi_C, "D": D}


def hybrid_por_score(
    params: Dict[str, Any], question: str, history: List["QaRecord"], *, w1: float = 0.6, w2: float = 0.4
) -> float:
    """Return PoR score based on UGHer model and semantic similarity."""
    trig = por_trigger(params["q"], params["s"], params["t"], params["phi_C"], params["D"])
    por_score = float(trig["score"])
    d_val = float(params["D"])
    por_model = por_score * (1 - d_val)
    if history:
        max_sim = max(_similarity(question, h.question) for h in history)
        por_sim = 1.0 - max_sim
    else:
        por_sim = 1.0
    return round(w1 * por_model + w2 * por_sim, 3)


def delta_e(prev_answer: str | None, curr_answer: str) -> float:
    """Return ΔE based on semantic difference between answers."""
    if prev_answer is None:
        return 0.0
    return round(1.0 - _similarity(prev_answer, curr_answer), 3)


def grv_score(answer: str) -> float:
    """Proxy to :func:`core.grv.grv_score`."""
    return _grv_score(answer)


def evaluate_metrics(por: float, delta_e_val: float, grv: float) -> Tuple[float, bool]:
    """Return overall score and adoption flag."""
    score = W_POR * por + W_DE * (1 - delta_e_val) + W_GRV * grv
    return round(score, 3), score >= ADOPT_TH


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class QaRecord:
    question: str
    answer: str
    por: float
    delta_e: float
    grv: float
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Cycle logic
# ---------------------------------------------------------------------------

def run_cycle(steps: int, output: Path, *, interactive: bool = False) -> None:
    """Run the Q&A cycle for ``steps`` iterations and store results to ``output``."""
    history: List[QaRecord] = []
    prev_answer: str | None = None

    for idx in range(steps):
        if interactive:
            question = input(f"質問{idx + 1}: ")
        else:
            question = f"Q{idx + 1}"

        answer = get_ai_response(question)
        params = estimate_ugh_params(question, history)
        por = hybrid_por_score(params, question, history)
        de = delta_e(prev_answer, answer)
        grv = grv_score(answer)

        print(f"[AI応答] {answer}")
        print(f"【PoR】{por:.2f} | 【ΔE】{de:.3f} | 【grv】{grv:.3f}")
        score, adopted = evaluate_metrics(por, de, grv)
        decision = "採用" if adopted else "不採用"
        print(f"【総合】{score:.3f} → {decision}")

        if adopted:
            history.append(QaRecord(question, answer, por, de, grv))
        prev_answer = answer

    if history:
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(history[0]).keys()))
            writer.writeheader()
            for rec in history:
                writer.writerow(asdict(rec))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    """Command line interface for the collector."""
    parser = argparse.ArgumentParser(description="PoR/ΔE/grv collector")
    parser.add_argument("-n", "--steps", type=int, default=10, help="number of cycles")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("por_history.csv"), help="CSV output path"
    )
    parser.add_argument("--auto", action="store_true", help="run without interactive prompts")
    args = parser.parse_args(argv)

    run_cycle(args.steps, args.output, interactive=not args.auto)


if __name__ == "__main__":
    main()
