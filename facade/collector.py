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
import sys

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
# weight factors inside hybrid_por_score
POR_W1: float = 0.6
POR_W2: float = 0.4


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _dummy_response(question: str) -> str:
    """Return a deterministic fallback response."""
    return f"Answer for '{question}'"


def _call_openai(
    question: str, *, temperature: float | None = None, max_tokens: int | None = None
) -> str:
    """Return a response from OpenAI's API using the v1 ``Client`` interface."""

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
        params: Dict[str, Any] = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": question}],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        resp: Any = client.chat.completions.create(**params)
        return str(resp.choices[0].message.content).strip()
    except Exception as err:
        # In case of failure, return a beginner-friendly error message
        return f"OpenAI APIの呼び出しに失敗しました: {err}"


def _call_anthropic(
    question: str, *, temperature: float | None = None, max_tokens: int | None = None
) -> str:
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
        params: Dict[str, Any] = {
            "model": "claude-3-opus-20240229",
            "messages": [{"role": "user", "content": question}],
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        resp: Any = client.messages.create(**params)
        return str(resp.content[0].text).strip()
    except Exception as err:
        return f"Anthropic APIの呼び出しに失敗しました: {err}"


def _call_gemini(
    question: str, *, temperature: float | None = None, max_tokens: int | None = None
) -> str:
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
        gen_cfg: Dict[str, Any] = {}
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if max_tokens is not None:
            gen_cfg["max_output_tokens"] = max_tokens
        resp: Any = model.generate_content(question, generation_config=gen_cfg or None)
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


def generate_next_question(prev_answer: str, history: List["HistoryEntry"], provider: str) -> str:
    """Return the next question using the specified generator."""
    if provider == "template":
        from typing import cast
        from secl.qa_cycle import (
            simulate_generate_next_question_from_answer,
            HistoryEntry as SeHistoryEntry,
        )
        q, _ = simulate_generate_next_question_from_answer(
            prev_answer, cast(List[SeHistoryEntry], history)
        )
        return q
    prompt = (
        f"あなたは研究用データ収集AIです。前の回答「{prev_answer}」を踏まえ、関連しつつも"
        "テーマや視点を少し変えた新しい質問を1文生成してください。"
    )
    if provider == "openai":
        return _call_openai(prompt, temperature=1.2, max_tokens=40)
    if provider == "anthropic":
        return _call_anthropic(prompt, temperature=1.2, max_tokens=40)
    if provider == "gemini":
        return _call_gemini(prompt, temperature=1.2, max_tokens=40)
    return _dummy_response(prompt)


# ---------------------------------------------------------------------------
# Metric calculations
# ---------------------------------------------------------------------------

def estimate_ugh_params(question: str, history: List["HistoryEntry"]) -> Dict[str, float]:
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
    params: Dict[str, float],
    question: str,
    history: List["HistoryEntry"],
    *,
    w1: float | None = None,
    w2: float | None = None,
) -> float:
    """Return PoR score based on UGHer model and semantic similarity."""
    if w1 is None:
        w1 = POR_W1
    if w2 is None:
        w2 = POR_W2
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


def grv_score(answer: str, *, mode: str = "simple") -> float:
    """Proxy to :func:`core.grv.grv_score`."""
    return _grv_score(answer, mode=mode)


def evaluate_metrics(por: float, delta_e_val: float, grv: float) -> Tuple[float, bool]:
    """Return overall score and adoption flag."""
    score = W_POR * por + W_DE * (1 - delta_e_val) + W_GRV * grv
    return round(score, 3), score >= ADOPT_TH


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    question: str
    answer: str
    por: float
    delta_e: float
    grv: float
    timestamp: float = field(default_factory=time.time)

# --- 互換エイリアス ------------
QARecord = HistoryEntry
# --------------------------------


# ---------------------------------------------------------------------------
# Cycle logic
# ---------------------------------------------------------------------------

def run_cycle(
    steps: int,
    output: Path,
    *,
    interactive: bool = False,
    stdin_mode: bool = False,
    quiet: bool = False,
    summary: bool = False,
    jsonl_path: Path | None = None,
    q_provider: str = "template",
    grv_mode: str = "simple",
) -> None:
    """Run the Q&A cycle for ``steps`` iterations and store results.

    Parameters
    ----------
    steps : int
        Number of iterations to execute.
    output : Path
        CSV file path for history output.
    jsonl_path : Path | None, optional
        When provided, adopted records are also appended to this JSONL file.
    q_provider : str, optional
        Question generation provider (LLM or template).
    grv_mode : str, optional
        Mode for :func:`core.grv.grv_score`.
    """
    history: List[HistoryEntry] = []
    prev_answer: str | None = None
    executed = 0

    # optional progress bar
    try:
        from tqdm import tqdm  # type: ignore
        iter_range = tqdm(range(steps), disable=quiet)
    except Exception:
        iter_range = range(steps)

    provider = os.getenv("AI_PROVIDER", "dummy")
    q_prov = q_provider

    for idx in iter_range:
        if stdin_mode:
            question = sys.stdin.readline().strip()
            if not question:
                break
        elif interactive:
            question = input(f"質問{idx + 1}: ")
        else:
            question = generate_next_question(prev_answer or "", history, q_prov)

        answer = get_ai_response(question)
        params = estimate_ugh_params(question, history)
        por = hybrid_por_score(params, question, history)
        de = delta_e(prev_answer, answer)
        grv = grv_score(answer, mode=grv_mode)

        if not quiet:
            print(f"[AI応答] {answer}")
        print(f"【PoR】{por:.2f} | 【ΔE】{de:.3f} | 【grv】{grv:.3f}")
        score, adopted = evaluate_metrics(por, de, grv)
        if not quiet:
            decision = "採用" if adopted else "不採用"
            print(f"【総合】{score:.3f} → {decision}")

        if adopted:
            history.append(HistoryEntry(question, answer, por, de, grv))
            if jsonl_path is not None:
                rec_dict = {
                    "question": question,
                    "answer": answer,
                    "por": por,
                    "delta_e": de,
                    "grv": grv,
                    "provider": provider,
                }
                with open(jsonl_path, "a", encoding="utf-8") as jfh:
                    jfh.write(f"{rec_dict}\n")
        prev_answer = answer
        executed += 1

    if history:
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(history[0]).keys()))
            writer.writeheader()
            for rec in history:
                writer.writerow(asdict(rec))

    if summary:
        por_avg = sum(h.por for h in history) / len(history) if history else 0.0
        de_avg = sum(h.delta_e for h in history) / len(history) if history else 0.0
        grv_avg = sum(h.grv for h in history) / len(history) if history else 0.0
        print("=== Summary ===")
        print(f"count:  {len(history)} / {executed}  (adopted / total)")
        print(f"PoR μ:  {por_avg:.2f}")
        print(f"ΔE μ:   {de_avg:.2f}")
        print(f"grv μ:  {grv_avg:.2f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    """Command line interface for the collector."""
    parser = argparse.ArgumentParser(description="PoR/ΔE/grv collector")
    parser.add_argument("-n", "--steps", type=int, default=10, help="number of cycles")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("por_history.csv"),
        help="CSV output path",
    )
    parser.add_argument("--auto", action="store_true", help="run without interactive prompts")
    parser.add_argument("--w-por", type=float, help="override W_POR")
    parser.add_argument("--w-de", type=float, help="override W_DE")
    parser.add_argument("--w-grv", type=float, help="override W_GRV")
    parser.add_argument("--adopt-th", type=float, help="override ADOPT_TH")
    parser.add_argument("--por-w1", type=float, help="hybrid_por_score w1")
    parser.add_argument("--por-w2", type=float, help="hybrid_por_score w2")
    parser.add_argument("--quiet", action="store_true", help="suppress AI responses")
    parser.add_argument("--stdin", action="store_true", help="consume questions from stdin")
    parser.add_argument("--summary", action="store_true", help="print summary at end")
    parser.add_argument("--exp-id", type=str, help="experiment id for output directory")
    parser.add_argument("--jsonl", action="store_true", help="also write JSONL")
    parser.add_argument(
        "--q-provider",
        choices=["openai", "anthropic", "gemini", "template"],
        default="template",
        help="question generation provider",
    )
    parser.add_argument(
        "--grv-mode",
        choices=["simple", "entropy"],
        default="simple",
        help="grv score mode",
    )

    args = parser.parse_args(argv)

    if args.exp_id:
        exp_id = args.exp_id
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        exp_id = f"{ts}_q-{args.q_provider}_g-{args.grv_mode}"
    output_dir = Path("runs") / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / (args.output.name if isinstance(args.output, Path) else "por_history.csv")
    output_jsonl = output_dir / "por_history.jsonl" if args.jsonl else None

    if args.w_por is not None:
        globals()["W_POR"] = args.w_por
    if args.w_de is not None:
        globals()["W_DE"] = args.w_de
    if args.w_grv is not None:
        globals()["W_GRV"] = args.w_grv
    if args.adopt_th is not None:
        globals()["ADOPT_TH"] = args.adopt_th
    if args.por_w1 is not None:
        globals()["POR_W1"] = args.por_w1
    if args.por_w2 is not None:
        globals()["POR_W2"] = args.por_w2

    run_cycle(
        args.steps,
        output_csv,
        interactive=(not args.auto and not args.stdin),
        stdin_mode=args.stdin,
        quiet=args.quiet,
        summary=args.summary,
        jsonl_path=output_jsonl,
        q_provider=args.q_provider,
        grv_mode=args.grv_mode,
    )

# LLM質問+拡張ジャンプで300問収集
# python facade/collector.py --auto -n 300 \
#     --q-provider openai --grv-mode entropy \
#     --quiet --summary


if __name__ == "__main__":
    main()
