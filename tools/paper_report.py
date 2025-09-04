from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from tools.stats_common import attach_meta, pick_col as _pick
from tools.stats_advanced import corr_stats, ols_standardized, series_summary


def _pick_col(df: pd.DataFrame, cands: Tuple[str, ...]) -> Optional[str]:
    return _pick(df, cands)


def _hash_q(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def compute_row_stats(df: pd.DataFrame) -> Dict[str, Any]:
    por = _pick_col(df, ("por", "PoR"))
    de = _pick_col(df, ("delta_e", "AE", "de", "delta_e_"))
    grv = _pick_col(df, ("grv", "grv_mu", "grv_mu"))
    dom = _pick_col(df, ("domain", "domains"))
    dif = _pick_col(df, ("difficulty", "difficulties"))
    n = int(len(df))
    zero_de = int((df[de] == 0).sum()) if de else 0
    return {
        "rows": n,
        "domains": int(df[dom].nunique()) if dom else 0,
        "difficulties": int(df[dif].nunique()) if dif else 0,
        "por_mu": float(df[por].mean()) if por else 0.0,
        "ae_mu": float(df[de].mean()) if de else 0.0,
        "grv_mu": float(df[grv].mean()) if grv else 0.0,
        "zero_de": zero_de,
    }


def to_question_level(df: pd.DataFrame) -> pd.DataFrame:
    col_q = _pick_col(df, ("question", "prompt", "input"))
    col_dom = _pick_col(df, ("domain", "domains"))
    col_dif = _pick_col(df, ("difficulty", "difficulties"))
    col_por = _pick_col(df, ("por", "PoR"))
    col_de = _pick_col(df, ("delta_e", "AE", "de", "delta_e_"))
    col_grv = _pick_col(df, ("grv", "grv_mu", "grv_mu"))
    if not col_q:
        raise SystemExit("question column not found")
    qkey = df[col_q].astype(str).map(_hash_q)
    agg = (
        df.assign(__qkey=qkey)
        .groupby("__qkey", as_index=False)
        .agg(
            {
                col_q: "first",
                col_dom: "first" if col_dom else "size",
                col_dif: "first" if col_dif else "size",
                col_por: "mean" if col_por else "size",
                col_de: "mean" if col_de else "size",
                col_grv: "mean" if col_grv else "size",
            }
        )
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
    return agg.rename(columns=rename)


def compute_q_stats(dfq: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(dfq))
    return {
        "questions": n,
        "domains": int(dfq["domain"].nunique()) if "domain" in dfq.columns else 0,
        "difficulties": int(dfq["difficulty"].nunique()) if "difficulty" in dfq.columns else 0,
        "por_mu_q": float(dfq["por"].mean()) if "por" in dfq.columns else 0.0,
        "ae_mu_q": float(dfq["delta_e"].mean()) if "delta_e" in dfq.columns else 0.0,
        "grv_mu_q": float(dfq["grv"].mean()) if "grv" in dfq.columns else 0.0,
    }


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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_md(path: Path, payload: Dict[str, Any]) -> None:
    s = payload
    rl, ql = s["row_level"], s["question_level"]
    lines: list[str] = []
    lines += ["# UGH Dataset — Paper Report", ""]
    lines += ["## Totals", ""]
    lines += [f"- Rows (adopted): **{rl['rows']}** / Zero-ΔE rows: **{rl['zero_de']}**"]
    lines += [f"- Questions: **{ql['questions']}**"]
    lines += [f"- Domains: rows **{rl['domains']}**, questions **{ql['domains']}**"]
    lines += [f"- Difficulties: rows **{rl['difficulties']}**, questions **{ql['difficulties']}**", ""]
    lines += ["## Means (row-level)", ""]
    lines += [f"- PoR μ: **{rl['por_mu']:.3f}**, ΔE μ: **{rl['ae_mu']:.3f}**, grv μ: **{rl['grv_mu']:.3f}**", ""]
    lines += ["## Means (question-level)", ""]
    lines += [
        f"- PoR μ (Q): **{ql['por_mu_q']:.3f}**, ΔE μ (Q): **{ql['ae_mu_q']:.3f}**, grv μ (Q): **{ql['grv_mu_q']:.3f}**",
        "",
    ]

    meta = s.get("meta")
    if meta:
        lines += ["## Meta", ""]
        lines += [f"- csv: `{meta.get('csv', '')}`"]
        c = meta.get("counts")
        if c:
            pre = c.get("records_total", "?")
            kept = c.get("kept", "?")
            zero = c.get("zeros_removed", "?")
            lines += [f"- rows (pre → post): **{pre} → {kept}** (zero-ΔE removed: {zero})"]
        lines += [f"- date: {meta.get('date', '')}", ""]

    dist = s.get("distributions", {})
    if dist:
        lines += ["## Distributions (row & question)", ""]
        lines += [
            "| level | metric | n | mean | median | IQR | 95% CI (mean) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for lvl_key in ("row", "question"):
            lvl = dist.get(lvl_key, {})
            for mkey, summ in lvl.items():
                if not summ:
                    continue
                n = summ.get("n")
                mean = summ.get("mean")
                med = summ.get("median")
                iqr = summ.get("iqr")
                ci = summ.get("ci95") or [None, None]
                lines += [
                    f"| {lvl_key} | {mkey} | {n} | {mean:.3f} | {med:.3f} | {iqr:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |"
                ]
        lines += [""]

    cor = s.get("correlation", {})
    reg = s.get("regression", {})
    if cor:
        lines += ["## Correlation (question-level)", ""]
        if "por_grv" in cor:
            pr = cor["por_grv"]["pearson"]
            sp = cor["por_grv"]["spearman"]
            lines += [
                f"- Pearson(PoR, grv): r={pr.get('r'):.3f} (p={pr.get('p') if pr.get('p') is not None else 'n/a'})"
            ]
            lines += [
                f"- Spearman(PoR, grv): ρ={sp.get('rho'):.3f} (p={sp.get('p') if sp.get('p') is not None else 'n/a'})"
            ]
        if "por_de" in cor:
            pr = cor["por_de"]["pearson"]
            sp = cor["por_de"]["spearman"]
            lines += [
                f"- Pearson(PoR, ΔE): r={pr.get('r'):.3f} (p={pr.get('p') if pr.get('p') is not None else 'n/a'})"
            ]
            lines += [
                f"- Spearman(PoR, ΔE): ρ={sp.get('rho'):.3f} (p={sp.get('p') if sp.get('p') is not None else 'n/a'})"
            ]
        lines += [""]

    if reg:
        lines += ["## Regression (standardized OLS on question-level): PoR ~ grv + ΔE", ""]
        r2 = reg.get("r2")
        adj = reg.get("adj_r2")
        r2_txt = f"{r2:.3f}" if r2 is not None else "n/a"
        adj_txt = f"{adj:.3f}" if adj is not None else "n/a"
        lines += [f"- R²={r2_txt}  adj.R²={adj_txt}", ""]
        coef = reg.get("coef", {})
        se = reg.get("stderr", {})
        t = reg.get("t", {})
        p = reg.get("p", {})
        lines += ["| term | beta | se | t | p |", "|---|---:|---:|---:|---:|"]
        for term in reg.get("design", []):
            lines += [
                f"| {term} | {coef.get(term, float('nan')):.3f} | {se.get(term, float('nan')):.3f} | {t.get(term, float('nan')):.3f} | {p.get(term, 'n/a')} |"
            ]
        lines += [""]

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

    try:
        import matplotlib
        import matplotlib.pyplot as plt  # type: ignore

        matplotlib.use("Agg")
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        cols = [("por", "PoR"), ("delta_e", "AE"), ("grv", "grv")]
        for ax, (c, label) in zip(axes, cols):
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


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build paper-ready report from dataset.csv")
    p.add_argument("--csv", required=True, help="Input dataset CSV (after filtering)")
    p.add_argument("--outdir", required=True, help="Output directory (e.g., reports/DATE)")
    p.add_argument("--meta", required=False, default=None, help="Optional meta.json path (to show pre/post counts).")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    row_stats = compute_row_stats(df)
    dfq = to_question_level(df)
    q_stats = compute_q_stats(dfq)

    dist_row: Dict[str, Any] = {}
    col_por = _pick_col(df, ("por", "PoR"))
    col_de = _pick_col(df, ("delta_e", "AE", "de", "delta_e_"))
    col_grv = _pick_col(df, ("grv", "grv_mu", "grv_mu"))
    if col_por:
        dist_row["por"] = series_summary(df[col_por])
    if col_de:
        dist_row["delta_e"] = series_summary(df[col_de])
    if col_grv:
        dist_row["grv"] = series_summary(df[col_grv])

    dist_q: Dict[str, Any] = {}
    if "por" in dfq.columns:
        dist_q["por"] = series_summary(dfq["por"])
    if "delta_e" in dfq.columns:
        dist_q["delta_e"] = series_summary(dfq["delta_e"])
    if "grv" in dfq.columns:
        dist_q["grv"] = series_summary(dfq["grv"])

    corr_payload: Dict[str, Any] = {}
    if "por" in dfq.columns and "grv" in dfq.columns:
        corr_payload["por_grv"] = corr_stats(dfq["por"], dfq["grv"])
    if "por" in dfq.columns and "delta_e" in dfq.columns:
        corr_payload["por_de"] = corr_stats(dfq["por"], dfq["delta_e"])

    reg_payload: Dict[str, Any] = {}
    xcols = [c for c in ("grv", "delta_e") if c in dfq.columns]
    if "por" in dfq.columns and xcols:
        reg_payload = ols_standardized(dfq, "por", xcols, add_intercept=True, standardize=True)

    payload: Dict[str, Any] = {
        "row_level": row_stats,
        "question_level": q_stats,
        "by_domain_q": by_category(dfq, "domain"),
        "by_difficulty_q": by_category(dfq, "difficulty"),
        "distributions": {"row": dist_row, "question": dist_q},
        "correlation": corr_payload,
        "regression": reg_payload,
        "question_df_cols": {k: dfq[k].tolist() for k in ("por", "delta_e", "grv")
                             if k in dfq.columns},
    }

    payload = attach_meta(payload, csv=args.csv, meta_path=args.meta)

    outdir = Path(args.outdir)
    write_json(outdir / "paper_stats.json", payload)
    write_md(outdir / "paper_report.md", payload)
    print(json.dumps({"row": row_stats, "q": q_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

