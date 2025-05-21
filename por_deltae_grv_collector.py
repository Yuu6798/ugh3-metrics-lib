#!/usr/bin/env python3
"""Lightweight PoR/ΔE/grv collector utility."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Any
import random
from por_trigger import por_trigger

import grv_scoring


# ---------------------------------------------------------------------------
# basic metric helpers
# ---------------------------------------------------------------------------

def get_ai_response(question: str) -> str:
    """Return a deterministic AI response for the given question."""
    return f"Answer for '{question}'"


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """Return similarity ratio between two texts."""
    return SequenceMatcher(None, text1, text2).ratio()


def estimate_ugh_params(question: str, history: List["QaRecord"]) -> Dict[str, Any]:
    """Estimate UGH parameters from the question text and history length."""
    q_len = len(question)
    h_len = len(history)

    q = min(1.0, q_len / 50.0)
    s = min(1.0, 0.5 + 0.05 * h_len)
    t = 0.5 + min(0.5, q_len / 100.0)
    phi_C = round(0.8 + random.uniform(-0.05, 0.05), 3)
    D = round(min(0.5, 0.1 + 0.02 * h_len) + random.uniform(0, 0.05), 3)

    return {"q": q, "s": s, "t": t, "phi_C": phi_C, "D": D}


def hybrid_por_score(params: Dict[str, Any], question: str, history: List["QaRecord"], w1: float = 0.6, w2: float = 0.4) -> float:
    """Return a hybrid PoR score using UGHer and semantic similarity."""
    trig = por_trigger(params["q"], params["s"], params["t"], params["phi_C"], params["D"])
    por1 = trig["score"] * (1 - params["D"])

    if history:
        max_sim = max(compute_semantic_similarity(question, h.question) for h in history)
        por2 = 1.0 - max_sim
    else:
        por2 = 1.0

    score = w1 * por1 + w2 * por2
    return round(score, 2)


def deltae_score(prev_answer: str | None, curr_answer: str) -> float:
    """Compute ΔE between two answers based on textual difference."""
    if prev_answer is None:
        return 0.0
    diff = 1.0 - compute_semantic_similarity(prev_answer, curr_answer)
    return round(diff, 3)


def grv_score(answer: str) -> float:
    """Wrapper around :func:`grv_scoring.grv_score`."""
    return grv_scoring.grv_score(answer)


# ---------------------------------------------------------------------------
# record structure
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
# cycle logic
# ---------------------------------------------------------------------------

def run_cycle(steps: int, output: Path, interactive: bool = False) -> None:
    """Run a Q&A cycle and persist metrics to ``output`` CSV."""
    history: List[QaRecord] = []
    prev_answer: str | None = None

    for step in range(steps):
        if interactive:
            question = input(f"質問{step + 1}: ")
        else:
            question = f"Q{step + 1}"

        answer = get_ai_response(question)
        params = estimate_ugh_params(question, history)
        por = hybrid_por_score(params, question, history)
        delta_e = deltae_score(prev_answer, answer)
        grv = grv_score(answer)

        print(f"[AI応答] {answer}")
        print(f"【PoR】{por:.2f} | 【ΔE】{delta_e:.3f} | 【grv】{grv:.3f}")

        history.append(QaRecord(question, answer, por, delta_e, grv))
        prev_answer = answer

    if history:
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(history[0]).keys()))
            writer.writeheader()
            for rec in history:
                writer.writerow(asdict(rec))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PoR/ΔE/grv collector")
    parser.add_argument("-n", "--steps", type=int, default=10, help="number of cycles")
    parser.add_argument("-o", "--output", type=Path, default=Path("por_history.csv"), help="CSV output path")
    parser.add_argument("--auto", action="store_true", help="run without interactive prompts")
    args = parser.parse_args(argv)

    run_cycle(args.steps, args.output, interactive=not args.auto)


if __name__ == "__main__":
    main()
