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
import itertools
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import logging

from ugh.adapters.metrics import DeltaE4, GrvV4, PorV4, prefetch_embed_model
from core.history_entry import HistoryEntry
from utils.config_loader import CONFIG
from facade.secl_hook import maybe_apply_secl

LOGGER = logging.getLogger(__name__)

# --- helpers for LLM config (env) ---
def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    import os
    val = os.getenv(key)
    return val if (val is not None and str(val).strip() != "") else default

STOPWORDS: set[str] = set()
_stop_path = Path(__file__).resolve().parent.parent / "data" / "jp_stop.txt"
try:
    with open(_stop_path, "r", encoding="utf-8") as sfh:
        STOPWORDS.update(word.strip() for word in sfh if word.strip())
except Exception:  # pragma: no cover - optional dependency
    pass

DOMAINS: list[str] = ["general", "creative", "technical", "specialized"]
DIFFICULTIES: list[int] = [1, 2, 3, 4, 5]


def stratified_pairs(n: int) -> list[tuple[str, int]]:
    """Return ``n`` domain/difficulty pairs with roughly equal coverage."""
    base = list(itertools.product(DOMAINS, DIFFICULTIES))
    q, r = divmod(n, len(base))
    pairs = base * q + random.sample(base, r) if r else base * q
    random.shuffle(pairs)
    return pairs


# --- metric singletons ----------------------------------------------------
_POR = PorV4()  # PoR v4
try:
    _DE = DeltaE4()
except RuntimeError as err:  # pragma: no cover - network/setup failure
    print(f"[ERROR] {err}", file=sys.stderr)
    sys.exit(2)
_GRV = GrvV4()  # grv v4

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
    """Return a deterministic fallback response (last resort)."""
    return f"Answer for '{question}'"




def _call_openai(
    question: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    role: str | None = None,
) -> str:
    """Return an answer from OpenAI. Prefer v1 Client; fall back to v1 module API."""
    # NOTE: `role` is informational only at the moment.

    # --- read configuration from env
    api_key = _env("OPENAI_API_KEY")
    model = _env("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    system = (_env("OPENAI_SYSTEM", "") or "").strip()

    # --- build messages payload
    msgs: List[Dict[str, str]] = [{"role": "user", "content": question}]
    if system:
        msgs = [{"role": "system", "content": system}] + msgs

    # --- primary path: v1 Client
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        client_kwargs: Dict[str, Any] = {"model": model, "messages": msgs}
        if temperature is not None:
            client_kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            client_kwargs["max_tokens"] = int(max_tokens)
        client_resp: Any = client.chat.completions.create(**client_kwargs)
        return (client_resp.choices[0].message.content or "").strip()
    except Exception as exc:
        LOGGER.warning("[AI provider] openai v1 Client not available; trying module API. (%s)", exc)

    # --- fallback path: v1 module-level API (no v0 symbols)
    try:
        import openai  # type: ignore[import-not-found]
        mod_kwargs: Dict[str, Any] = {"model": model, "messages": msgs}
        if temperature is not None:
            mod_kwargs["temperature"] = float(temperature)
        if max_tokens is not None:
            mod_kwargs["max_tokens"] = int(max_tokens)
        mod_resp: Any = openai.chat.completions.create(**mod_kwargs)  # type: ignore[attr-defined]
        return (mod_resp.choices[0].message.content or "").strip()
    except Exception as exc:
        LOGGER.warning("[AI provider] openai v1 module call failed: %s; using dummy.", exc)
        return _dummy_response(question)


def _call_anthropic(
    question: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    role: str | None = None,
) -> str:
    """Return an answer from Anthropic's API or a friendly error message.

    The ``role`` argument is informational only and currently unused.
    """
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
    question: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    role: str | None = None,
) -> str:
    """Return an answer from Google Gemini or a friendly error message.

    The ``role`` argument is informational only and currently unused.
    """
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
    """Return an AI-generated answer using the chosen provider.

    Parameters
    ----------
    question : str
        User question to send to the AI.
    provider : str | None, optional
        Which AI service to use (``openai``, ``anthropic``, ``gemini``, ``dummy``).
        If ``None``, the environment variable ``AI_PROVIDER`` will be checked.
        If no provider is found, a simple deterministic answer is returned.

    Notes
    -----
    To add support for another provider, implement a new ``_call_*`` function
    similar to the above examples and extend the mapping in this function.

    OpenAI API access uses ``openai>=1.0.0`` with the ``Client`` class. See
    the project README for development installation instructions.
    """

    raw_provider: str = provider or os.getenv("AI_PROVIDER") or "dummy"
    prov = raw_provider.lower()
    print(f"[AI provider] {prov}")

    if prov == "openai":
        return _call_openai(question, role="answer")
    if prov == "anthropic":
        return _call_anthropic(question, role="answer")
    if prov == "gemini":
        return _call_gemini(question, role="answer")
    if prov == "dummy":
        return _dummy_response(question)

    print(f"[error] unsupported AI_PROVIDER '{prov}'")
    return f"未対応のAI_PROVIDER '{prov}' が指定されました。"


def generate_next_question(
    prev_answer: str,
    history: List["HistoryEntry"],
    provider: str,
    domain: str,
    difficulty: int,
) -> str:
    """Return the next question using the specified LLM provider."""
    if provider == "template":
        from secl.qa_cycle import simulate_generate_next_question_from_answer

        q, _ = simulate_generate_next_question_from_answer(prev_answer, history)
        return q
    prompt = (
        f"あなたは研究用データ収集AIです。ドメイン『{domain}』で難易度{difficulty}"
        f"の質問を作成します。前の回答「{prev_answer}」を踏まえ、関連しつつも"
        "テーマや視点を少し変えた新しい質問を1文生成してください。"
    )
    if provider == "openai":
        return _call_openai(prompt, temperature=1.2, max_tokens=40, role="question")
    if provider == "anthropic":
        return _call_anthropic(prompt, temperature=1.2, max_tokens=40, role="question")
    if provider == "gemini":
        return _call_gemini(prompt, temperature=1.2, max_tokens=40, role="question")
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


def por_score(question: str, hist: list["HistoryEntry"]) -> float:
    """Return PoR score via the v4 metric."""
    params = estimate_ugh_params(question, hist)
    return float(_POR.score(question, hist, params=params))  # type: ignore[arg-type, call-arg]


def delta_e(prev_answer: str | None, curr_answer: str) -> float:
    """Return ΔE score via the v4 metric."""
    if prev_answer is None:
        return 0.0
    return float(_DE.score(prev_answer, curr_answer))


def grv_score(answer: str, *, mode: str = "simple") -> float:
    """Return grv score via the v4 metric."""
    _ = mode
    return float(_GRV.score(answer, ""))


def evaluate_metrics(por: float, delta_e_val: float, grv: float) -> Tuple[float, bool]:
    """Return overall score and adoption flag."""
    score = W_POR * por + W_DE * (1 - delta_e_val) + W_GRV * grv
    return round(score, 3), score >= ADOPT_TH


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


# ``HistoryEntry`` is imported from :mod:`core.history_entry` for reuse across
# modules.
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
    q_provider: str = "openai",
    ai_provider: str = "openai",
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
        Question generation provider (LLM or ``dummy``).
    ai_provider : str, optional
        Provider used for answer generation.
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

    provider = ai_provider or os.getenv("AI_PROVIDER", "dummy")
    q_prov = q_provider
    dd_pairs = stratified_pairs(steps)

    for idx, (domain, difficulty) in zip(iter_range, dd_pairs):
        if stdin_mode:
            question = sys.stdin.readline().strip()
            if not question:
                break
        elif interactive:
            question = input(f"質問{idx + 1}: ")
        else:
            question = generate_next_question(prev_answer or "", history, q_prov, domain, difficulty)

        answer = get_ai_response(question, provider=provider)
        por = por_score(question, history)
        from core.delta_e_v4 import delta_e_v4
        de = delta_e_v4(prev_answer or "", answer)
        if de == 0.0:
            print("[WARN] \u0394E could not be computed; check inputs.")
        grv = grv_score(answer, mode=grv_mode)

        if not quiet:
            print(f"[AI応答] {answer}")
        print(f"【PoR】{por:.2f} | 【ΔE】{de:.3f} | 【grv】{grv:.3f}")
        score, adopted = evaluate_metrics(por, de, grv)
        if not quiet:
            decision = "採用" if adopted else "不採用"
            print(f"【総合】{score:.3f} → {decision}")

        if adopted:
            history.append(
                HistoryEntry(
                    question=question,
                    answer_a=prev_answer or "",
                    answer_b=answer,
                    por=por,
                    delta_e=de,
                    grv=grv,
                    domain=domain,
                    difficulty=difficulty,
                )
            )
            if jsonl_path is not None:
                rec_dict = {
                    "question": question,
                    "answer_a": prev_answer or "",
                    "answer_b": answer,
                    "por": por,
                    "delta_e": de,
                    "grv": grv,
                    "domain": domain,
                    "difficulty": difficulty,
                    "timestamp": history[-1].timestamp,
                    "score": history[-1].score,
                    "spike": history[-1].spike,
                    "external": history[-1].external,
                    "anomaly_por": history[-1].anomaly_por,
                    "anomaly_delta_e": history[-1].anomaly_delta_e,
                    "anomaly_grv": history[-1].anomaly_grv,
                    "por_null": history[-1].por_null,
                    "score_threshold": history[-1].score_threshold,
                    "provider": provider,
                }
                with open(jsonl_path, "a", encoding="utf-8") as jfh:
                    jfh.write(f"{rec_dict}\n")
        # SECL を実行し、戻り値の履歴を常にマージ（jump 有無に関わらず）
        _secl_res = maybe_apply_secl(question, history, CONFIG)
        if _secl_res is not None:
            history = _secl_res.updated_history

        # Always advance the cycle state regardless of jump decision.
        prev_answer = answer
        executed += 1

    if history:
        fields = [
            "question",
            "answer_a",
            "answer_b",
            "por",
            "delta_e",
            "grv",
            "domain",
            "difficulty",
            "timestamp",
            "score",
            "spike",
            "external",
            "anomaly_por",
            "anomaly_delta_e",
            "anomaly_grv",
            "por_null",
            "score_threshold",
        ]
        with open(output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for rec in history:
                writer.writerow({f: getattr(rec, f) for f in fields})

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


def main(argv: List[str] | None = None) -> int:
    """Command line interface for the collector."""
    # Preload embedding model once
    try:
        prefetch_embed_model()
    except Exception:
        if os.getenv("DELTAE4_FALLBACK", "").lower() not in ("1", "true", "yes", "hash"):
            raise

    parser = argparse.ArgumentParser(description="PoR/ΔE/grv collector (ばらつき付き自動生成)")
    parser.add_argument(
        "-n",
        "--steps",
        type=int,
        default=50,
        help="number of cycles with stratified domain/difficulty (default: 50)",
    )
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
        choices=["openai", "anthropic", "gemini", "dummy", "template"],
        default="openai",
        help="question generation provider",
    )
    parser.add_argument(
        "--ai-provider",
        choices=["openai", "anthropic", "gemini", "dummy"],
        default="openai",
        help="answer generation provider",
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
        exp_id = f"{ts}_q-{args.q_provider}_a-{args.ai_provider}_g-{args.grv_mode}"
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
        ai_provider=args.ai_provider,
        grv_mode=args.grv_mode,
    )

    # ここまで来て CSV を出力できていれば “成果物あり” として正常終了にする。
    # 採択数不足や ΔE=0 の混入は後続ステップ（top-off / ビルド / アップロード）が処理するため非エラー扱い。
    return 0


# LLM同士で自動進化対話（質問も応答もOpenAI）
# python facade/collector.py --auto -n 50 --q-provider openai --ai-provider openai --quiet --summary

# 質問はGemini、応答はOpenAIで異種AI対話も可
# python facade/collector.py --auto -n 50 --q-provider gemini --ai-provider openai --quiet --summary


if __name__ == "__main__":
    import sys
    sys.exit(main())
