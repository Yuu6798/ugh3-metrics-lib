#!/usr/bin/env python3
"""Simple PoR/ΔE/grv data collection script.

This CLI utility runs a SECL-like Q&A cycle and records metrics for
PoR score, ΔE (existence energy change), and grv (vocabulary gravity).
The formulas exactly replicate those defined in ``secl_qa_cycle.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration if present; return empty defaults otherwise."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


CONFIG = load_config()
GRV_WINDOW: int = CONFIG.get("GRV_WINDOW", 10)


@dataclass
class QaEntry:
    """Container for one Q&A interaction."""

    question: str
    answer: str
    por_score: float
    delta_e: float
    grv: float
    timestamp: float = field(default_factory=time.time)
    por_null: bool = False


# ---------------------------------------------------------------------------
# metric calculations
# ---------------------------------------------------------------------------

def novelty_score(question: str, history: List[QaEntry]) -> float:
    """Return novelty score based on maximum similarity to history."""
    if not history:
        return 1.0
    max_sim = max(
        SequenceMatcher(None, question, entry.question).ratio()
        for entry in history
    )
    penalty = 0.5 * max_sim
    return max(0.0, 1.0 - penalty)


def simulate_delta_e(current_q: str, next_q: str, answer: str) -> float:
    """Compute ΔE following secl_qa_cycle implementation."""
    q2q = abs(len(current_q) - len(next_q)) / 30.0
    q2a = 1.0 - SequenceMatcher(None, current_q, answer).ratio()
    rand = random.uniform(0.1, 0.8)
    delta_e_jump = min(1.0, 0.5 * q2q + 0.3 * q2a + 0.2 * rand)
    return round(delta_e_jump, 3)


def calc_grv_field(history: List[QaEntry], window: int = GRV_WINDOW) -> float:
    """Calculate vocabulary gravity using recent entries."""
    recent = history[-window:] if len(history) >= window else history
    vocab: set[str] = set()
    for entry in recent:
        vocab |= set(entry.question.split())
        vocab |= set(entry.answer.split())
    grv_val = min(1.0, len(vocab) / 30.0)
    return round(grv_val, 3)


def por_score_from(novelty: float, delta_e: float) -> float:
    """Combine novelty and ΔE into a PoR score."""
    score = 0.6 * novelty + 0.4 * delta_e
    return round(score, 2)


def is_grv_stagnation(history: List[float], window: int = 3, threshold: float = 0.05) -> bool:
    """Return True if grv change is below threshold over the window."""
    if len(history) < window + 1:
        return False
    diffs = [abs(history[-i] - history[-i - 1]) for i in range(1, window + 1)]
    return sum(diffs) / window < threshold


def detect_por_null(q: str, ans: str, novelty: float, delta_e: float) -> bool:
    """Detect PoR Null state when inputs are empty or metrics zero."""
    if not q or not ans:
        return True
    return novelty == 0 and delta_e == 0


# ---------------------------------------------------------------------------
# simple helpers for automated Q&A generation
# ---------------------------------------------------------------------------


def generate_answer(question: str) -> str:
    """Return a canned answer string for a given question."""
    templates = [
        f"'{question}'ですね。考察すると、…",
        f"その問い、'{question}'への応答例は…",
        f"'{question}'をめぐって考えたいのは…",
    ]
    return random.choice(templates) + "（AI応答続き）"


def generate_next_question(answer: str) -> str:
    """Derive the next question from an answer with a random ID."""
    base = f"「{answer}」を受けて、次に考えるべき具体的な論点は？"
    return f"{base}#{random.randint(100, 999)}"


def save_history_to_csv(path: Path, history: List[QaEntry]) -> None:
    """Persist Q&A history to a CSV file."""
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for entry in history:
            writer.writerow(asdict(entry))


# ---------------------------------------------------------------------------
# main cycle logic
# ---------------------------------------------------------------------------


def run_cycle(steps: int, output: Path) -> None:
    """Execute Q&A steps and record metrics."""
    history: List[QaEntry] = []
    grv_hist: List[float] = []

    current_q = "意識はどこから生まれるか？"
    prev_q = current_q

    for step in range(steps):
        print(f"\n--- Step {step + 1} ---")
        ans = generate_answer(current_q)

        temp_entry = QaEntry(current_q, ans, 0, 0, 0)
        grv = calc_grv_field(history + [temp_entry])
        grv_hist.append(grv)

        if step == 0:
            delta_e = 0.0
        else:
            delta_e = simulate_delta_e(prev_q, current_q, ans)

        novelty = novelty_score(current_q, history)
        score = por_score_from(novelty, delta_e)
        por_null = detect_por_null(current_q, ans, novelty, delta_e)

        history.append(
            QaEntry(
                question=current_q,
                answer=ans,
                por_score=score,
                delta_e=delta_e,
                grv=grv,
                por_null=por_null,
            )
        )

        prev_q = current_q
        current_q = generate_next_question(ans)

        if is_grv_stagnation(grv_hist):
            print("[grv停滞検知] 新語彙導入が停滞しています。")

        print(f"PoR {score:.2f} / ΔE {delta_e:.3f} / grv {grv:.3f}")

    save_history_to_csv(output, history)
    print(f"\nSaved history to {output}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PoR/ΔE/grv data collector")
    parser.add_argument("-n", "--steps", type=int, default=10, help="number of cycles")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("por_history.csv"), help="CSV output path"
    )
    args = parser.parse_args()
    run_cycle(args.steps, args.output)


if __name__ == "__main__":
    main()
