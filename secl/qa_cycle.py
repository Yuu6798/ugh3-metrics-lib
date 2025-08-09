"""SECL Q&A Cycle Simulation with Metrics Logging and Anomaly Handling.

Improvement Plan:
- Separate functions with type hints and documentation.
- Parameters loaded from external ``config.json`` for easy tuning.
- Metrics (PoR score, ΔE, grv) recorded to history and optionally persisted to CSV.
- Summary reporting helper for evolution and anomaly detection output.
- Metrics computation isolated for future UGH (Unconscious Gravity Hypothesis) integration.
- PoR/ΔE/grv anomaly detection with optional HTTP alerts.
- Automatic history backup with configurable interval.
- Basic error handling and unit tests provided.
- Security note: avoid executing untrusted input and sanitize file paths when extending.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils.config_loader import CONFIG as _CONFIG
from core.history_entry import HistoryEntry
from ugh.adapters.metrics import (
    compute_por,
    compute_delta_e_embed,
    compute_grv_window,
)

CONFIG: Dict[str, Any] = _CONFIG
__all__ = ["HistoryEntry", "CONFIG", "main_qa_cycle"]

MAX_LOG_SIZE: int = CONFIG.get("MAX_LOG_SIZE", 10)
BASE_SCORE_THRESHOLD: float = CONFIG.get("BASE_SCORE_THRESHOLD", 0.5)
DELTA_E_WINDOW: int = CONFIG.get("DELTA_E_WINDOW", 5)
DELTA_E_UP: float = CONFIG.get("DELTA_E_UP", 0.1)
DELTA_E_DOWN: float = CONFIG.get("DELTA_E_DOWN", 0.1)
DELTA_E_HIGH: float = CONFIG.get("DELTA_E_HIGH", 0.7)
DELTA_E_LOW: float = CONFIG.get("DELTA_E_LOW", 0.3)
GRV_WINDOW: int = CONFIG.get("GRV_WINDOW", 3)
GRV_STAGNATION_TH: float = CONFIG.get("GRV_STAGNATION_TH", 0.05)
DUPLICATE_THRESHOLD: float = CONFIG.get("DUPLICATE_THRESHOLD", 0.9)
EPS_BASE: float = CONFIG.get("EPS_BASE", 0.2)
LOW_POR_TH: float = CONFIG.get("LOW_POR_TH", 0.25)
# ΔE (delta_e) の高しきい値はここで一元管理
HIGH_DELTA_TH: float = CONFIG.get("HIGH_DELTA_TH", 0.85)
ANOMALY_POR_THRESHOLD: float = CONFIG.get("ANOMALY_POR_THRESHOLD", 0.9)
ANOMALY_DELTA_E_THRESHOLD: float = CONFIG.get("ANOMALY_DELTA_E_THRESHOLD", 0.9)
ANOMALY_GRV_THRESHOLD: float = CONFIG.get("ANOMALY_GRV_THRESHOLD", 0.95)
BACKUP_DIR: str = CONFIG.get("BACKUP_DIR", "backups")
ALT_BACKUP_DIR: str = CONFIG.get("ALT_BACKUP_DIR", "backups_alt")
BACKUP_INTERVAL: int = CONFIG.get("BACKUP_INTERVAL", 10)
ALERT_POST_URL: str | None = CONFIG.get("ALERT_POST_URL")



def novelty_score(question: str, history_list: List[HistoryEntry]) -> float:
    if not history_list:
        return 1.0
    max_similarity = max(SequenceMatcher(None, question, entry.question).ratio() for entry in history_list)
    penalty = 0.5 * max_similarity
    return max(0.0, 1.0 - penalty)


def is_duplicate_question(question: str, history_list: List[HistoryEntry]) -> bool:
    for entry in history_list:
        if SequenceMatcher(None, question, entry.question).ratio() >= DUPLICATE_THRESHOLD:
            return True
    return False


def simulate_delta_e(current_q: str, next_q: str, answer: str) -> float:
    q2q = abs(len(current_q) - len(next_q)) / 30.0
    q2a = 1.0 - SequenceMatcher(None, current_q, answer).ratio()
    random_factor = random.uniform(0.1, 0.8)
    delta_e_jump = min(1.0, 0.5 * q2q + 0.3 * q2a + 0.2 * random_factor)
    return round(delta_e_jump, 3)


def simulate_generate_answer(question: str) -> str:
    templates = [
        f"'{question}'ですね。考察すると、…",
        f"その問い、'{question}'への応答例は…",
        f"'{question}'をめぐって考えたいのは…",
    ]
    return random.choice(templates) + "（AI応答続き）"


def simulate_external_knowledge() -> str:
    external_concepts = [
        "新たな理論",
        "異なる文化",
        "未解明の現象",
        "未来技術",
        "社会進化",
        "宇宙論",
    ]
    return f"[{random.choice(external_concepts)}]"


def calc_grv_field(history_list: List[HistoryEntry], window: int = GRV_WINDOW) -> Tuple[float, set[str]]:
    recent = history_list[-window:] if len(history_list) >= window else history_list
    vocab_set: set[str] = set()
    for entry in recent:
        vocab_set |= set(entry.question.split())
        vocab_set |= set(entry.answer_b.split())
    grv = min(1.0, len(vocab_set) / 30.0)
    return round(grv, 3), vocab_set


def is_grv_stagnation(grv_history: List[float], window: int = GRV_WINDOW, threshold: float = GRV_STAGNATION_TH) -> bool:
    if len(grv_history) < window + 1:
        return False
    diffs = [abs(grv_history[-i] - grv_history[-i - 1]) for i in range(1, window + 1)]
    return sum(diffs) / window < threshold


def update_score_threshold(
    delta_e_history: List[float],
    base_threshold: float = BASE_SCORE_THRESHOLD,
    window: int = DELTA_E_WINDOW,
    up: float = DELTA_E_UP,
    down: float = DELTA_E_DOWN,
) -> float:
    if len(delta_e_history) < window:
        return base_threshold
    avg_delta_e = sum(delta_e_history[-window:]) / window
    if avg_delta_e > DELTA_E_HIGH:
        return min(1.0, base_threshold + up)
    if avg_delta_e < DELTA_E_LOW:
        return max(0.0, base_threshold - down)
    return base_threshold


def simulate_grv_gain_with_jump(current_state: Dict[str, Any], base: str = "ジャンプ") -> float:
    grv_val: float = float(current_state["grv"])
    base_vocab = set(current_state["vocab_set"])
    added = {base + str(random.randint(100, 999))}
    simulated = base_vocab | added
    gain = min(1.0, len(simulated) / 30.0) - grv_val
    return gain


def simulate_grv_gain_with_external_info(current_state: Dict[str, Any]) -> float:
    grv_val: float = float(current_state["grv"])
    base_vocab = set(current_state["vocab_set"])
    added = {simulate_external_knowledge()}
    simulated = base_vocab | added
    gain = min(1.0, len(simulated) / 30.0) - grv_val
    return gain


def select_action_for_jump(state: Dict[str, Any]) -> str:
    """Return "jump" or "external" based on stagnation signals."""
    if state.get("low_por"):
        return "external"
    if state.get("high_delta"):
        return "jump"
    if state.get("stagnate_grv"):
        return random.choice(["jump", "external"])
    return "jump"


def record_to_log(history_list: List[HistoryEntry], entry: HistoryEntry) -> List[HistoryEntry]:
    if len(history_list) >= MAX_LOG_SIZE:
        min_idx = min(range(len(history_list)), key=lambda i: history_list[i].score)
        history_list.pop(min_idx)
    history_list.append(entry)
    return history_list


def detect_spike(current_score: float, history_list: List[HistoryEntry]) -> bool:
    if len(history_list) < 1:
        return False
    prev_score = float(history_list[-1].score)
    return current_score - prev_score > 0.3


def save_history_to_csv(path: Path, history_list: List[HistoryEntry]) -> None:
    try:
        with open(path, "w", newline="", encoding="utf-8") as fh:
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
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for entry in history_list:
                writer.writerow({f: getattr(entry, f) for f in fields})
    except Exception as exc:
        print(f"[Error] Failed to save history: {exc}")


def save_history_to_json(path: Path, history_list: List[HistoryEntry]) -> None:
    """Persist history as JSON for easier external analysis."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([asdict(h) for h in history_list], fh, ensure_ascii=False, indent=2)
    except Exception as exc:  # pragma: no cover - simple print
        print(f"[Error] Failed to save history JSON: {exc}")


def send_alert(message: str) -> None:
    """Send anomaly alert via HTTP POST if configured."""
    print(f"[Alert] {message}")
    if ALERT_POST_URL:
        try:
            import urllib.request

            data = json.dumps({"message": message}).encode("utf-8")
            req = urllib.request.Request(
                ALERT_POST_URL,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5):
                pass
        except Exception as exc:  # pragma: no cover - network issues
            print(f"[Alert Error] {exc}")


def check_metric_anomalies(por: float, delta_e: float, grv: float) -> Tuple[bool, bool, bool]:
    """Return flags for PoR, ΔE and grv anomalies based on thresholds."""
    por_flag = por >= ANOMALY_POR_THRESHOLD
    delta_flag = delta_e >= ANOMALY_DELTA_E_THRESHOLD
    grv_flag = grv >= ANOMALY_GRV_THRESHOLD
    if por_flag:
        send_alert(f"PoR anomaly detected: score {por:.2f}")
    if delta_flag:
        send_alert(f"ΔE anomaly detected: {delta_e:.2f}")
    if grv_flag:
        send_alert(f"grv anomaly detected: {grv:.2f}")
    return por_flag, delta_flag, grv_flag


def detect_por_null(next_q: str, answer: str, novelty: float, delta_e: float) -> bool:
    """Detect PoR Null when metrics cannot be computed or output is empty."""
    if not next_q or not answer:
        return True
    return novelty == 0 and delta_e == 0


def backup_history(directory: Path, history_list: List[HistoryEntry], prefix: str) -> None:
    """Save a JSON backup with timestamp. Attempt alternate dir on failure."""
    timestamp = int(time.time())
    filename = f"{prefix}_{timestamp}.json"
    path = directory / filename
    try:
        directory.mkdir(parents=True, exist_ok=True)
        save_history_to_json(path, history_list)
    except Exception as exc:
        print(f"[Backup Error] {exc} -> trying alternate dir")
        alt = Path(ALT_BACKUP_DIR)
        alt.mkdir(parents=True, exist_ok=True)
        save_history_to_json(alt / filename, history_list)


def print_log_summary(history_list: List[HistoryEntry]) -> None:
    print("---- 履歴サマリー ----")
    if not history_list:
        print("（履歴なし）")
        return
    for i, entry in enumerate(history_list):
        q_disp = entry.question[:32] + "..." if len(entry.question) > 32 else entry.question
        bar = "█" * int(entry.score * 10)
        anomaly = entry.anomaly_por or entry.anomaly_delta_e or entry.anomaly_grv
        flags = (
            f"{'S' if entry.spike else ' '}|"
            f"{'E' if entry.external else ' '}|"
            f"{'A' if anomaly else ' '}|"
            f"{'N' if entry.por_null else ' '}"
        )
        print(f"{i+1:02d}.Q:'{q_disp}'|S:{entry.score:.2f}|ΔE:{entry.delta_e:.2f}|Grv:{entry.grv:.2f}|{flags} {bar}")
    scores = [e.score for e in history_list]
    grvs = [e.grv for e in history_list]
    print(f"[統計] 平均S:{sum(scores)/len(scores):.2f}, 平均Grv:{sum(grvs)/len(grvs):.2f}")
    print("-----------------------------------")


def simulate_generate_next_question_from_answer(
    answer: str, history_list: List[HistoryEntry], epsilon: float = EPS_BASE
) -> Tuple[str, bool]:
    ext_flag = False
    if random.random() < epsilon:
        ext_flag = True
        return f"（新視点）{simulate_external_knowledge()}についての問いは？#{random.randint(1000, 9999)}", ext_flag
    base_question = f"「{answer}」を受けて、次に考えるべき具体的な論点は？"
    if random.random() < 0.2:
        ext_flag = True
        base_question = f"「{answer}」＋{simulate_external_knowledge()}の観点で何が課題か？"
    return f"{base_question}#{random.randint(100, 999)}", ext_flag


def simulate_learn(
    question: str,
    answer: str,
    score: float,
    delta_e: float,
    grv: float,
    history_list: List[HistoryEntry],
    spike_flag: bool,
    ext_flag: bool,
    score_threshold: float,
    anomaly_flags: Tuple[bool, bool, bool],
    por_null_flag: bool,
) -> List[HistoryEntry]:
    """Record learning step with anomaly flags and dynamic score threshold."""
    if is_duplicate_question(question, history_list):
        print("  [重複検知] ほぼ同じ問いが履歴に存在。採用せずスキップ。")
        return history_list
    if score >= score_threshold:
        entry = HistoryEntry(
            question=question,
            answer_a=history_list[-1].answer_b if history_list else "",
            answer_b=answer,
            por=score,
            delta_e=delta_e,
            grv=grv,
            domain="general",
            difficulty=1,
        )
        entry.score = score
        entry.spike = spike_flag
        entry.external = ext_flag
        entry.anomaly_por = anomaly_flags[0]
        entry.anomaly_delta_e = anomaly_flags[1]
        entry.anomaly_grv = anomaly_flags[2]
        entry.por_null = por_null_flag
        entry.score_threshold = score_threshold
        updated = record_to_log(history_list, entry)
        print(f"  [採用] スコア {score:.2f} で履歴数: {len(updated)}")
        if spike_flag:
            print(f"  >>> [PoRスパイク!] 急上昇: {score:.2f}")
        return updated
    print(f"  [不採用] スコア {score:.2f} で履歴に追加せず。")
    return history_list


def main_qa_cycle(n_steps: int = 25, save_path: Path | None = None) -> List[HistoryEntry]:
    print("=== 進化型SECL Q&Aサイクル（統合版） ===")
    print(f"--- 設定: {n_steps}ステップ, 履歴MAX:{MAX_LOG_SIZE} ---")
    history_list: List[HistoryEntry] = []
    delta_e_history: List[float] = []
    grv_history: List[float] = []
    # defaults to avoid undefined references on first iteration
    score: float = 0.0
    score_threshold: float = BASE_SCORE_THRESHOLD
    low_por_th = CONFIG.get("LOW_POR_TH", LOW_POR_TH)
    jump_cooldown = 0
    current_question = "意識はどこから生まれるか？"
    prev_question: str = current_question
    for step in range(n_steps):
        print(f"\n--- Step {step+1} ---")
        jump_cooldown = max(0, jump_cooldown - 1)
        answer = simulate_generate_answer(current_question)
        temp_entry = HistoryEntry(
            question=current_question,
            answer_a=history_list[-1].answer_b if history_list else "",
            answer_b=answer,
            por=0.0,
            delta_e=0.0,
            grv=0.0,
            domain="general",
            difficulty=1,
        )
        temp_entry.score = 0.0
        temp_entry.spike = False
        temp_entry.external = False
        grv, vocab_set = compute_grv_window(history_list + [temp_entry])
        grv_history.append(grv)
        if step == 0:
            delta_e = 0.0
        else:
            delta_e = compute_delta_e_embed(prev_question, current_question, answer)
        delta_e_history.append(delta_e)
        por = compute_por(current_question, answer)
        score = round(por, 3)
        score_threshold = update_score_threshold(delta_e_history, BASE_SCORE_THRESHOLD)
        score_threshold = max(score_threshold, low_por_th)
        stagnate_grv = is_grv_stagnation(grv_history)
        low_por = por < low_por_th
        # 高いデルタEを基準にした値
        high_delta = delta_e >= HIGH_DELTA_TH
        stagnation = low_por or high_delta or stagnate_grv
        if stagnation and step > 0 and jump_cooldown == 0:
            print("【再学習】高いデルタEのジャンプ外部入力判定中...")
            state = {
                "low_por": low_por,
                "high_delta": high_delta,
                "stagnate_grv": stagnate_grv,
                "por": por,
                "score_threshold": score_threshold,
                "low_por_th": low_por_th,
                "high_delta_th": HIGH_DELTA_TH,
            }
            action = select_action_for_jump(state)
            if action == "jump":
                next_question = (
                    f"(ジャンプ){current_question}＋{random.choice(['変革','未知','分岐'])}#{random.randint(1000,9999)}"
                )
                ext_flag = False
                jump_cooldown = 3
            else:
                next_question = f"(外部注入){simulate_external_knowledge()}#{random.randint(1000,9999)}"
                ext_flag = True
                jump_cooldown = 3
        else:
            next_question, ext_flag = simulate_generate_next_question_from_answer(
                answer, history_list, epsilon=EPS_BASE
            )
        novelty = novelty_score(next_question, history_list)
        spike_flag = detect_spike(score, history_list)
        anomaly_flags = check_metric_anomalies(por, delta_e, grv)
        por_null_flag = detect_por_null(next_question, answer, novelty, delta_e)
        history_list = simulate_learn(
            next_question,
            answer,
            score,
            delta_e,
            grv,
            history_list,
            spike_flag,
            ext_flag,
            score_threshold,
            anomaly_flags,
            por_null_flag,
        )
        if (step + 1) % BACKUP_INTERVAL == 0:
            backup_history(Path(BACKUP_DIR), history_list, f"step{step+1}")
        print_log_summary(history_list)
        prev_question = current_question
        current_question = next_question
    print("\n=== 最終学習履歴 ===")
    print_log_summary(history_list)
    if save_path and history_list:
        save_history_to_csv(save_path, history_list)
        save_history_to_json(save_path.with_suffix(".json"), history_list)
    backup_history(Path(BACKUP_DIR), history_list, "final")
    return history_list


if __name__ == "__main__":
    main_qa_cycle(25, Path("history.csv"))
