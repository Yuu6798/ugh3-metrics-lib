from __future__ import annotations
import argparse, json, os, hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

# --- helpers -----------------------------------------------------------------
def _pick_col(df: pd.DataFrame, cands: Tuple[str, ...]) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
    return None


def _hash_q(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

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
    # 必須っぽい列を柔軟に拾う
    col_q  = _pick_col(df, ("question","prompt","input"))
    col_a  = _pick_col(df, ("answer","answer_a","output"))
    col_dom= _pick_col(df, ("domain","domains"))
    col_dif= _pick_col(df, ("difficulty","difficulties"))
    col_por= _pick_col(df, ("por","PoR"))
    col_de = _pick_col(df, ("delta_e","ΔE","de","delta_e_"))
    col_grv= _pick_col(df, ("grv","grv_μ","grv_mu"))
    if not col_q:
        raise SystemExit("question column not found")
    # 質問キー（質問文だけでハッシュ）: A/B を同じキーに
    qkey = df[col_q].astype(str).map(_hash_q)
    g = df.assign(__qkey=qkey).groupby("__qkey", as_index=False)
    # A/B の平均で代表値を作る（必要なら max などにも変更可）
    agg = g.agg({
        col_q: "first",
        col_dom: "first" if col_dom else "size",
        col_dif: "first" if col_dif else "size",
        col_por: "mean" if col_por else "size",
        col_de:  "mean" if col_de  else "size",
        col_grv: "mean" if col_grv else "size",
    })
    # 列名の標準化
    rename = {}
    if col_q:   rename[col_q]  = "question"
    if col_dom: rename[col_dom]= "domain"
    if col_dif: rename[col_dif]= "difficulty"
    if col_por: rename[col_por]= "por"
    if col_de:  rename[col_de] = "delta_e"
    if col_grv: rename[col_grv]= "grv"
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

def by_category(df: pd.DataFrame, key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if key not in df.columns: return out
    for k, g in df.groupby(key):
        out[str(k)] = {
            "count": int(len(g)),
            "por_mu": float(g["por"].mean()) if "por" in g.columns else 0.0,
            "ae_mu":  float(g["delta_e"].mean()) if "delta_e" in g.columns else 0.0,
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
    lines += [f"- Difficulties: rows **{rl['difficulties']}**, questions **{ql['difficulties']}**", ""]
    lines += ["## Means (row-level)", ""]
    lines += [f"- PoR μ: **{rl['por_mu']:.3f}**, ΔE μ: **{rl['ae_mu']:.3f}**, grv μ: **{rl['grv_mu']:.3f}**", ""]
    lines += ["## Means (question-level)", ""]
    lines += [f"- PoR μ (Q): **{ql['por_mu_q']:.3f}**, ΔE μ (Q): **{ql['ae_mu_q']:.3f}**, grv μ (Q): **{ql['grv_mu_q']:.3f}**", ""]
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
        from math import ceil
        # hist of PoR / ΔE / grv on question-level
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        cols = [("por","PoR"), ("delta_e","ΔE"), ("grv","grv")]
        for ax,(c,label) in zip(axes, cols):
            if c in s["question_df_cols"]:
                ax.hist(s["question_df_cols"][c], bins=20)
                ax.set_title(label)
        fig.tight_layout()
        img = path.parent / "hist_question.png"
        fig.savefig(img, dpi=160)
        lines += ["## Histograms (question-level)", "", f"![hist_question](hist_question.png)", ""]
    except Exception as e:
        lines += ["_Note: matplotlib not available; skipped plots._", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --- CLI ----------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build paper-ready report from dataset.csv")
    p.add_argument("--csv", required=True, help="Input dataset CSV (after filtering)")
    p.add_argument("--outdir", required=True, help="Output directory (e.g., reports/DATE)")
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
            k: dfq[k].tolist() for k in ("por","delta_e","grv") if k in dfq.columns
        },
    }
    outdir = Path(args.outdir)
    write_json(outdir / "paper_stats.json", payload)
    write_md(outdir / "paper_report.md", payload)
    print(json.dumps({"row": row_stats, "q": q_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
