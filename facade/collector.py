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
import os

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

def _dummy_response(question: str) -> str:
    """Return a deterministic fallback response."""
    return f"Answer for '{question}'"


def _call_openai(question: str) -> str:
    """Return an answer from OpenAI's API using the v1 ``Client`` interface.

    Requires ``openai>=1.0.0`` and the ``OPENAI_API_KEY`` environment variable.
    """

    # ``openai`` is optional. Install via `pip install openai>=1.0.0` if needed
    # and obtain an API key at https://platform.openai.com/account/api-keys
    try:
        # v1 では ``OpenAI`` クラスからクライアントを生成します
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        return "OpenAIプロバイダは利用できません（ライブラリ未インストール）"

    # Read the API key from the OPENAI_API_KEY environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        return "OpenAI APIキーが設定されていません。"

    # Create the client instance with the API key
    client = OpenAI(api_key=api_key)

    try:
        # Ask the model and return only the text part of the response
        resp: Any = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}],
        )
        return str(resp.choices[0].message.content).strip()
    except Exception as err:
        # In case of failure, return a beginner-friendly error message
        return f"OpenAI APIの呼び出しに失敗しました: {err}"


def _call_anthropic(question: str) -> str:
    """Return an answer from Anthropic's API or a friendly error message."""
    # anthropic library is optional; install via `pip install anthropic`
    # Get your API key from https://console.anthropic.com/ and set
    # ANTHROPIC_API_KEY in the environment.
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError:
        return "Anthropicプロバイダは利用できません（ライブラリ未インストール）"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        return "Anthropic APIキーが設定されていません。"

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp: Any = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=100,
            messages=[{"role": "user", "content": question}],
        )
        return str(resp.content[0].text).strip()
    except Exception as err:
        return f"Anthropic APIの呼び出しに失敗しました: {err}"


def _call_gemini(question: str) -> str:
    """Return an answer from Google Gemini or a friendly error message."""
    # google-generativeai library is optional; install via `pip install google-generativeai`
    # Get an API key at https://aistudio.google.com/app/apikey and set
    # GEMINI_API_KEY in your environment.
    try:
        from google.generativeai import configure, GenerativeModel  # type: ignore[import-not-found]
    except ImportError:
        return "Geminiプロバイダは利用できません（ライブラリ未インストール）"

    api_key = os.getenv("GEMINI_API_KEY")
    if not isinstance(api_key, str) or not api_key:
        return "Gemini APIキーが設定されていません。"

    configure(api_key=api_key)
    model = GenerativeModel("gemini-pro")
    try:
        resp: Any = model.generate_content(question)
        return str(resp.text).strip()
    except Exception as err:
        return f"Gemini APIの呼び出しに失敗しました: {err}"


def get_ai_response(question: str, provider: str | None = None) -> str:
    """Return an AI generated answer using the specified provider.

    Parameters
    ----------
    question : str
        User question to send to the AI.
    provider : str | None, optional
        Which AI service to use (``openai``, ``anthropic``, ``gemini``). If
        ``None``, the environment variable ``AI_PROVIDER`` will be checked.
        If no provider is found, a simple deterministic answer is returned.

    Notes
    -----
    To add support for another provider, implement a new ``_call_*`` function
    similar to the above examples and extend the mapping in this function.

    OpenAI API access uses ``openai>=1.0.0`` with the ``Client`` class. See
    ``requirements.txt`` for an installation example.
    """

    raw_provider: str = provider or os.getenv("AI_PROVIDER") or "dummy"
    prov = raw_provider.lower()
    print(f"[AI provider] {prov}")

    if prov == "openai":
        return _call_openai(question)
    if prov == "anthropic":
        return _call_anthropic(question)
    if prov == "gemini":
        return _call_gemini(question)
    if prov == "dummy":
        return _dummy_response(question)

    print(f"[error] unsupported AI_PROVIDER '{prov}'")
    return f"未対応のAI_PROVIDER '{prov}' が指定されました。"


def _similarity(text1: str, text2: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def estimate_ugh_params(question: str, history: List["QaRecord"]) -> Dict[str, float]:
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
    params: Dict[str, float], question: str, history: List["QaRecord"], *, w1: float = 0.6, w2: float = 0.4
) -> float:
    """Return PoR score based on UGHer model and semantic similarity."""
    trig = por_trigger(params["q"], params["s"], params["t"], params["phi_C"], params["D"])
    por_model = trig["score"] * (1 - params["D"])
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
