from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tools.stats_common import pick_col as _pick_col_impl, float_ as _float_helper, attach_meta


# --- helpers -----------------------------------------------------------------
def _pick_col(df: pd.DataFrame, cands: Tuple[str, ...]) -> Optional[str]:
    # delegate to shared helper (retain name for backward compatibility)
    return _pick_col_impl(df, cands)


def _hash_q(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _float(x: Any, default: float = 0.0) -> float:
    return _float_helper(x, default)


def _pick(v: Any, i: int, key: str) -> float:
    """Extract a numeric value from mapping/array-like ``v`` by label ``key`` or index ``i``.
    - dict-like: try label key
    - pandas.Series: prefer label (loc), else positional (iloc)
    - numpy.ndarray: positional by i (0-d uses item())
    - generic array-like: positional by i
    - otherwise: treat as scalar
    Non-scalar results are coerced to NaN; failures never raise.
    """

    try:
        val: Any

        # 1) dict-like (term -> value)
        if isinstance(v, dict):
            val = v.get(key, np.nan)

        # 2) pandas Series (label優先 → 位置)
        elif isinstance(v, pd.Series):
            if key in v.index:
                val = v.loc[key]
            elif 0 <= i < len(v):
                val = v.iloc[i]
            else:
                val = np.nan

        # 3) numpy ndarray（位置）
        elif isinstance(v, np.ndarray):
            if v.ndim == 0:
                val = v.item()
            elif 0 <= i < v.shape[0]:
                val = v[i]
            else:
                val = np.nan

        # 4) 配列風（__getitem__ と __len__ を持つ）
        elif hasattr(v, "__getitem__") and hasattr(v, "__len__"):
            val = v[i] if 0 <= i < len(v) else np.nan

        # 5) 上記以外はスカラー試行
        else:
            val = v

        # スカラー化：非スカラー（要素>1）は NaN にフォールバック
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            if arr.size == 1:
                return float(arr.reshape(()))
            return float("nan")
        except Exception:
            return float(val)
    except Exception:
        return float("nan")

# --- row-level stats (そのままの行 = A/B 含む) -----------------------------------
def compute_row_stats(df: pd.DataFrame) -> Dict[str, Any]:
    por = _pick_col(df, ("por","PoR"))
    de  = _pick_col(df, ("delta_e","ΔE","de","delta_e_"))
    grv = _pick_col(df, ("grv","grv_μ","grv_mu"))
    dom = _pick_col(df, ("domain","domains"))
    dif = _pick_col(df, ("difficulty","difficulties"))
    n = int(len(df))
    zero_de = int((df[de] == 0).sum()) if de else 0
    stats = {
        "rows": n,
        "domains": int(df[dom].nunique()) if dom else 0,
        "difficulties": int(df[dif].nunique()) if dif else 0,
        "por_mu": float(df[por].mean()) if por else 0.0,
        "ae_mu": float(df[de].mean()) if de else 0.0,
        "grv_mu": float(df[grv].mean()) if grv else 0.0,
        "zero_de": zero_de,
    }
    return stats

# --- question-level collapse (A/B を一本化) -----------------------------------
def to_question_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse A/B answers into single question-level entries."""
    col_q = _pick_col(df, ("question", "prompt", "input"))
    col_dom = _pick_col(df, ("domain", "domains"))
    col_dif = _pick_col(df, ("difficulty", "difficulties"))
    col_por = _pick_col(df, ("por", "PoR"))
    col_de = _pick_col(df, ("delta_e", "ΔE", "de", "delta_e_"))
    col_grv = _pick_col(df, ("grv", "grv_μ", "grv_mu"))
    if not col_q:
        raise SystemExit("question column not found")
    qkey = df[col_q].astype(str).map(_hash_q)
    g = df.assign(__qkey=qkey).groupby("__qkey", as_index=False)
    agg = g.agg(
        {
            col_q: "first",
            col_dom: "first" if col_dom else "size",
            col_dif: "first" if col_dif else "size",
            col_por: "mean" if col_por else "size",
            col_de: "mean" if col_de else "size",
            col_grv: "mean" if col_grv else "size",
        }
    )
    rename: Dict[str, str] = {}
    if col_q:
        rename[col_q] = "question"
    if col_dom:
        rename[col_dom] = "domain"
    if col_dif:
        rename[col_dif] = "difficulty"
    if col_por:
        rename[col_por] = "por"
    if col_de:
        rename[col_de] = "delta_e"
    if col_grv:
        rename[col_grv] = "grv"
    dfq = agg.rename(columns=rename)
    return dfq

def compute_q_stats(dfq: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(dfq))
    stats = {
        "questions": n,
        "domains": int(dfq["domain"].nunique()) if "domain" in dfq.columns else 0,
        "difficulties": int(dfq["difficulty"].nunique()) if "difficulty" in dfq.columns else 0,
        "por_mu_q": float(dfq["por"].mean()) if "por" in dfq.columns else 0.0,
        "ae_mu_q":  float(dfq["delta_e"].mean()) if "delta_e" in dfq.columns else 0.0,
        "grv_mu_q": float(dfq["grv"].mean()) if "grv" in dfq.columns else 0.0,
    }
    return stats

def by_category(dfq: pd.DataFrame, key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if key not in dfq.columns:
        return out
    for k, g in dfq.groupby(key):
        out[str(k)] = {
            "count": int(len(g)),
            "por_mu": float(g["por"].mean()) if "por" in g.columns else 0.0,
            "ae_mu": float(g["delta_e"].mean()) if "delta_e" in g.columns else 0.0,
            "grv_mu": float(g["grv"].mean()) if "grv" in g.columns else 0.0,
        }
    return dict(sorted(out.items(), key=lambda x: (-x[1]["count"], x[0])))

# --- writers ------------------------------------------------------------------
def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    s = payload
    rl, ql = s["row_level"], s["question_level"]
    lines = []
    lines += ["# UGH Dataset – Paper Report", ""]
    lines += ["## Totals", ""]
    lines += [f"- Rows (adopted): **{rl['rows']}** / Zero-ΔE rows: **{rl['zero_de']}**"]
    lines += [f"- Questions: **{ql['questions']}**"]
    lines += [f"- Domains: rows **{rl['domains']}**, questions **{ql['domains']}**"]
    lines += [
        f"- Difficulties: rows **{rl['difficulties']}**, questions **{ql['difficulties']}**",
        "",
    ]
    lines += ["## Means (row-level)", ""]
    lines += [f"- PoR μ: **{rl['por_mu']:.3f}**, ΔE μ: **{rl['ae_mu']:.3f}**, grv μ: **{rl['grv_mu']:.3f}**", ""]
    lines += ["## Means (question-level)", ""]
    lines += [
        f"- PoR μ (Q): **{ql['por_mu_q']:.3f}**, ΔE μ (Q): **{ql['ae_mu_q']:.3f}**, "
        f"grv μ (Q): **{ql['grv_mu_q']:.3f}**",
        "",
    ]
    meta = s.get("meta")
    if meta:
        lines += ["## Meta", ""]
        lines += [f"- csv: `{meta.get('csv', '')}`"]
        if "counts" in meta:
            c = meta["counts"]
            pre = c.get("records_total", "?")
            kept = c.get("kept", "?")
            zero = c.get("zeros_removed", "?")
            lines += [f"- rows (pre → post): **{pre} → {kept}** (zero-ΔE removed: {zero})"]
        lines += [f"- date: {meta.get('date', '')}", ""]
    reg = s.get("regression", {})
    if reg:
        lines += ["## Regression (standardized OLS on question-level): PoR ~ grv + ΔE", ""]
        r2 = reg.get("r2")
        adj = reg.get("adj_r2")
        r2_txt = f"{float(r2):.3f}" if isinstance(r2, (int, float)) else "n/a"
        adj_txt = f"{float(adj):.3f}" if isinstance(adj, (int, float)) else "n/a"
        lines += [f"- R²: {r2_txt}   adj.R²: {adj_txt}", ""]

        coef = reg.get("coef", {})
        se_container = reg.get("se", reg.get("stderr", {}))
        t_container = reg.get("t", reg.get("tstat", {}))
        p_any = reg.get("p", None)

        terms = reg.get("columns") or reg.get("colnames") or reg.get("design", [])
        if not terms and isinstance(coef, dict):
            terms = list(coef.keys())

        lines += ["| term | beta | se | t | p |",
                  "|:--|--:|--:|--:|--:|"]
        for i, term in enumerate(terms):
            beta_i = _pick(coef,        i, term)
            se_i   = _pick(se_container,i, term)
            t_i    = _pick(t_container, i, term)
            p_i: Optional[float] = None
            if isinstance(p_any, dict):
                p_i = p_any.get(term, None)
            elif isinstance(p_any, Sequence) and i < len(p_any):
                try:
                    p_i = float(p_any[i])
                except Exception:
                    p_i = None
            p_txt = f"{p_i:.3g}" if isinstance(p_i, (int, float)) else "n/a"

            lines += [f"| {term} | {beta_i:.3f} | {se_i:.3f} | {t_i:.3f} | {p_txt} |"]
        lines += [""]
    # domain/difficulty breakdown
    dom = s.get("by_domain_q", {})
    dif = s.get("by_difficulty_q", {})
    if dom:
        lines += ["## By Domain (question-level)", ""]
        lines += ["| domain | n | PoR μ | ΔE μ | grv μ |", "|---:|---:|---:|---:|---:|"]
        for k, v in dom.items():
            lines += [f"| {k} | {v['count']} | {v['por_mu']:.3f} | {v['ae_mu']:.3f} | {v['grv_mu']:.3f} |"]
        lines += [""]
    if dif:
        lines += ["## By Difficulty (question-level)", ""]
        lines += ["| difficulty | n | PoR μ | ΔE μ | grv μ |", "|---:|---:|---:|---:|---:|"]
        for k, v in dif.items():
            lines += [f"| {k} | {v['count']} | {v['por_mu']:.3f} | {v['ae_mu']:.3f} | {v['grv_mu']:.3f} |"]
        lines += [""]
    # optional plots
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        # hist of PoR / ΔE / grv on question-level
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        cols = [("por","PoR"), ("delta_e","ΔE"), ("grv","grv")]
        for ax,(c,label) in zip(axes, cols):
            if c in s["question_df_cols"]:
                ax.hist(s["question_df_cols"][c], bins=20)
                ax.set_title(label)
        fig.tight_layout()
        hist = path.parent / "hist_question.png"
        fig.savefig(hist, dpi=160)
        lines += ["## Histograms (question-level)", "", f"![]({hist.name})", ""]
    except Exception:
        lines += ["_Note: matplotlib not available; skipped plots._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --- CLI ----------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build paper-ready report from dataset.csv")
    p.add_argument("--csv", required=True, help="Input dataset CSV (after filtering)")
    p.add_argument("--outdir", required=True, help="Output directory (e.g., reports/DATE)")
    p.add_argument(
        "--meta", required=False, default=None, help="Optional meta.json path (to show pre/post counts)."
    )
    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)
    row_stats = compute_row_stats(df)
    dfq = to_question_level(df)
    q_stats = compute_q_stats(dfq)
    payload = {
        "row_level": row_stats,
        "question_level": q_stats,
        "by_domain_q": by_category(dfq, "domain"),
        "by_difficulty_q": by_category(dfq, "difficulty"),
        # for plotting (optional)
        "question_df_cols": {
            k: dfq[k].tolist() for k in ("por", "delta_e", "grv") if k in dfq.columns
        },
    }
    payload = attach_meta(payload, csv=args.csv, meta_path=args.meta)
    outdir = Path(args.outdir)
    write_json(outdir / "paper_stats.json", payload)
    write_md(outdir / "paper_report.md", payload)
    print(json.dumps({"row": row_stats, "q": q_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
