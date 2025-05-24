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
from math import log1p
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import os
import sys
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    # モジュール → None への再代入は型が食い違うので ignore
    yaml = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _EMBEDDER: Optional[SentenceTransformer] = None
except Exception:  # pragma: no cover - optional dependency may not be present
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore
    _EMBEDDER = None

from utils.config_loader import MAX_VOCAB_CAP

STOPWORDS: set[str] = set()
_stop_path = Path(__file__).resolve().parent.parent / "data" / "jp_stop.txt"
try:
    with open(_stop_path, "r", encoding="utf-8") as sfh:
        STOPWORDS.update(word.strip() for word in sfh if word.strip())
except Exception:  # pragma: no cover - optional dependency
    pass

from facade.trigger import por_trigger


def _load_embedder() -> Optional[SentenceTransformer]:
    """Lazy-load shared SentenceTransformer instance.

    Guarantees a concrete SentenceTransformer is returned.
    """
    global _EMBEDDER
    if _EMBEDDER is None and SentenceTransformer is not None:
        _EMBEDDER = SentenceTransformer("all-mpnet-base-v2")
    return _EMBEDDER

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

DEFAULT_CFG: Dict[str, float] = {
    "W_POR": 0.4,
    "W_DE": 0.4,
    "W_GRV": 0.2,
    "ADOPT_TH": 0.45,
    "POR_W1": 0.6,
    "POR_W2": 0.4,
}


def _load_yaml_cfg(path: Path) -> Dict[str, Any]:
    if yaml is None:
        return {}
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def merge_config(cfg: Dict[str, Any], args: argparse.Namespace | None = None) -> Dict[str, Any]:
    """Return runtime configuration merged from multiple sources."""
    result = cfg.copy()
    yaml_cfg = _load_yaml_cfg(Path(os.getenv("COLLECTOR_CONFIG", "config.yaml")))
    result.update({k: v for k, v in yaml_cfg.items() if k in result})
    for key in result:
        env_val = os.getenv(key)
        if env_val is not None:
            try:
                result[key] = float(env_val)
            except ValueError:
                pass
    if args is not None:
        mapping = {
            "w_por": "W_POR",
            "w_de": "W_DE",
            "w_grv": "W_GRV",
            "adopt_th": "ADOPT_TH",
            "por_w1": "POR_W1",
            "por_w2": "POR_W2",
        }
        for arg_key, cfg_key in mapping.items():
            val = getattr(args, arg_key, None)
            if val is not None:
                result[cfg_key] = val
    return result


CFG = DEFAULT_CFG.copy()


class Metrics:
    """Container for metric calculations."""

    def __init__(self, embedder: Optional[SentenceTransformer]) -> None:
        self.embedder = embedder

    def por_score(
        self,
        params: Dict[str, float],
        question: str,
        history: List["HistoryEntry"],
        *,
        w1: float | None = None,
        w2: float | None = None,
    ) -> float:
        if w1 is None:
            w1 = CFG["POR_W1"]
        if w2 is None:
            w2 = CFG["POR_W2"]
        trig = por_trigger(
            params["q"],
            params["s"],
            params["t"],
            params["phi_C"],
            params["D"],
        )
        por_model = trig["score"] * (1 - params["D"])
        if history:
            max_sim = max(_similarity(question, h.question) for h in history)
            por_sim = 1.0 - max_sim
        else:
            por_sim = 1.0
        return round(w1 * por_model + w2 * por_sim, 3)

    def delta_e(self, prev_answer: str | None, curr_answer: str) -> float:
        if prev_answer is None:
            return 0.0
        if self.embedder is None:
            try:
                self.embedder = _load_embedder()
            except Exception:
                print("[warn] embedding model load failed; falling back to length diff")
        if self.embedder is not None:
            try:
                v1 = self.embedder.encode(prev_answer)
                v2 = self.embedder.encode(curr_answer)
                num = float(np.dot(v1, v2))
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom == 0:
                    raise ValueError("zero norm")
                cos_sim = num / denom
                diff = max(0.0, 1.0 - cos_sim)
                return round(min(1.0, diff), 3)
            except Exception:
                pass
        diff = min(1.0, abs(len(prev_answer) - len(curr_answer)) / 50.0)
        return round(diff, 3)

    def grv(self, answer: str, *, mode: str = "simple") -> float:
        if isinstance(answer, str):
            tokens = [t for t in answer.split() if t not in STOPWORDS]
        else:
            tokens = [t for t in answer if t not in STOPWORDS]
        vocab = set(tokens)
        score = 0.0
        if vocab:
            score = log1p(len(vocab)) / log1p(MAX_VOCAB_CAP)
        return round(min(1.0, score), 3)


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def api_key_check(service: str, key: Optional[str]) -> Optional[str]:
    """Return ``None`` when the API key is valid, otherwise an error message."""
    if key is None or key == "":
        return f"{service} APIキーが設定されていません。"
    if not isinstance(key, str):
        return f"{service} APIキー形式が不正です。"
    return None

def _dummy_response(question: str) -> str:
    """Return a deterministic fallback response."""
    return f"Answer for '{question}'"


def _call_openai(
    question: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    role: str | None = None,
) -> str:
    """Return a response from OpenAI's API using the v1 ``Client`` interface.

    The ``role`` argument is informational only and currently unused.
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
    err = api_key_check("OpenAI", api_key)
    if err:
        return err

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
    err = api_key_check("Anthropic", api_key)
    if err:
        return err

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
    err = api_key_check("Gemini", api_key)
    if err:
        return err

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


AI_PROVIDERS: Dict[str, Any] = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "dummy": _dummy_response,
}


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
    ``requirements.txt`` for an installation example.
    """

    raw_provider: str = provider or os.getenv("AI_PROVIDER") or "dummy"
    prov = raw_provider.lower()
    print(f"[AI provider] {prov}")

    func = AI_PROVIDERS.get(prov)
    if func is not None:
        return func(question, role="answer")

    print(f"[error] unsupported AI_PROVIDER '{prov}'")
    return f"未対応のAI_PROVIDER '{prov}' が指定されました。"


def _similarity(text1: str, text2: str) -> float:
    """Return a similarity ratio between two strings."""
    return SequenceMatcher(None, text1, text2).ratio()


def generate_next_question(
    prev_answer: str, history: List["HistoryEntry"], provider: str
) -> str:
    """Return the next question using the specified LLM provider."""
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




def evaluate_metrics(por: float, delta_e_val: float, grv: float, cfg: Dict[str, Any]) -> Tuple[float, bool]:
    """Return overall score and adoption flag using ``cfg`` weights."""
    score = cfg["W_POR"] * por + cfg["W_DE"] * (1 - delta_e_val) + cfg["W_GRV"] * grv
    return round(score, 3), score >= cfg["ADOPT_TH"]


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
    cfg: Dict[str, Any] | None = None,
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
    if cfg is None:
        cfg = CFG
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
    metrics = Metrics(_load_embedder())

    for idx in iter_range:
        if stdin_mode:
            question = sys.stdin.readline().strip()
            if not question:
                break
        elif interactive:
            question = input(f"質問{idx + 1}: ")
        else:
            question = generate_next_question(prev_answer or "", history, q_prov)

        answer = get_ai_response(question, provider=provider)
        params = estimate_ugh_params(question, history)
        por = metrics.por_score(params, question, history)
        de = metrics.delta_e(prev_answer, answer)
        grv = metrics.grv(answer, mode=grv_mode)

        if not quiet:
            print(f"[AI応答] {answer}")
        print(f"【PoR】{por:.2f} | 【ΔE】{de:.3f} | 【grv】{grv:.3f}")
        score, adopted = evaluate_metrics(por, de, grv, cfg)
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

    global CFG
    CFG = merge_config(DEFAULT_CFG, args)

    if args.exp_id:
        exp_id = args.exp_id
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        exp_id = (
            f"{ts}_q-{args.q_provider}_a-{args.ai_provider}_"
            f"adth{CFG['ADOPT_TH']}_wpor{CFG['W_POR']}"
        )
    output_dir = Path("runs") / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / (args.output.name if isinstance(args.output, Path) else "por_history.csv")
    output_jsonl = output_dir / "por_history.jsonl" if args.jsonl else None

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
        cfg=CFG,
    )

# LLM同士で自動進化対話（質問も応答もOpenAI）
# python facade/collector.py --auto -n 50 --q-provider openai --ai-provider openai --quiet --summary

# 質問はGemini、応答はOpenAIで異種AI対話も可
# python facade/collector.py --auto -n 50 --q-provider gemini --ai-provider openai --quiet --summary


if __name__ == "__main__":
    main()
