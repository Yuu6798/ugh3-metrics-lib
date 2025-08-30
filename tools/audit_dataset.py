from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
from tools.stats_common import attach_meta


def _float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(df))
    domains = int(df["domain"].nunique()) if "domain" in df.columns else 0
    diffs = int(df["difficulty"].nunique()) if "difficulty" in df.columns else 0
    por_mu = _float(df["por"].mean()) if "por" in df.columns and n else 0.0
    ae_mu = _float(df["delta_e"].mean()) if "delta_e" in df.columns and n else 0.0
    grv_mu = _float(df["grv"].mean()) if "grv" in df.columns and n else 0.0
    zero_de = int((df["delta_e"] == 0).sum()) if "delta_e" in df.columns else 0
    return {
        "rows": n,
        "domains": domains,
        "difficulties": diffs,
        "por_mu": por_mu,
        "ae_mu": ae_mu,
        "grv_mu": grv_mu,
        "zero_de": zero_de,
    }


def judge(stats: Dict[str, Any], th: Dict[str, float]) -> Dict[str, Any]:
    """Return dict with pass/fail and each check result."""
    results = {
        "rows_ok": stats["rows"] >= th["min_rows"],
        "domains_ok": stats["domains"] >= th["min_domains"],
        "diffs_ok": stats["difficulties"] >= th["min_diffs"],
        "por_ok": stats["por_mu"] >= th["por_min"],
        "ae_ok": stats["ae_mu"] <= th["ae_max"],
        "grv_ok": stats["grv_mu"] >= th["grv_min"],
    }
    results["passed"] = all(results.values())
    return results


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_md(
    path: Path,
    stats: Dict[str, Any],
    results: Dict[str, Any],
    th: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Dataset Audit\n")
    lines.append("## Summary\n")
    lines.append(f"- rows (adopted): **{stats['rows']}**\n")
    lines.append(f"- domains (unique): **{stats['domains']}**\n")
    lines.append(f"- difficulties (unique): **{stats['difficulties']}**\n")
    lines.append(f"- PoR μ: **{stats['por_mu']:.3f}**\n")
    lines.append(f"- ΔE μ: **{stats['ae_mu']:.3f}**\n")
    lines.append(f"- grv μ: **{stats['grv_mu']:.3f}**\n")
    lines.append(f"- zero ΔE count: **{stats['zero_de']}**\n")
    lines.append("\n## Gate (thresholds)\n")
    lines.append(f"- min_rows ≥ {th['min_rows']} → {'✅' if results['rows_ok'] else '❌'}\n")
    lines.append(f"- min_domains ≥ {th['min_domains']} → {'✅' if results['domains_ok'] else '❌'}\n")
    lines.append(f"- min_diffs ≥ {th['min_diffs']} → {'✅' if results['diffs_ok'] else '❌'}\n")
    lines.append(f"- PoR μ ≥ {th['por_min']} → {'✅' if results['por_ok'] else '❌'}\n")
    lines.append(f"- ΔE μ ≤ {th['ae_max']} → {'✅' if results['ae_ok'] else '❌'}\n")
    lines.append(f"- grv μ ≥ {th['grv_min']} → {'✅' if results['grv_ok'] else '❌'}\n")
    lines.append(f"\n**Overall:** {'✅ PASS' if results['passed'] else '❌ FAIL'}\n")
    if meta:
        lines.append("\n## Meta\n")
        lines.append(f"- csv: `{meta.get('csv', '')}`")
        if "counts" in meta:
            c = meta["counts"]
            pre = c.get("records_total", "?")
            kept = c.get("kept", "?")
            zero = c.get("zeros_removed", "?")
            lines.append(
                f"- rows (pre → post): **{pre} → {kept}** (zero-ΔE removed: {zero})"
            )
        lines.append(f"- date: {meta.get('date', '')}\n")
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Audit dataset quality and gate.")
    p.add_argument("--csv", required=True, help="Input CSV path (adopted records).")
    p.add_argument("--json", required=True, help="Output audit JSON path.")
    p.add_argument("--md", required=True, help="Output audit Markdown path.")
    p.add_argument(
        "--meta",
        required=False,
        default=None,
        help="Optional meta.json path (to show pre/post counts).",
    )
    p.add_argument("--min-rows", type=int, default=40)
    p.add_argument("--min-domains", type=int, default=3)
    p.add_argument("--min-diffs", type=int, default=3)
    p.add_argument("--por-min", type=float, default=0.80)
    p.add_argument("--ae-max", type=float, default=0.30)
    p.add_argument("--grv-min", type=float, default=0.60)
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    stats = compute_stats(df)
    th = {
        "min_rows": args.min_rows,
        "min_domains": args.min_domains,
        "min_diffs": args.min_diffs,
        "por_min": args.por_min,
        "ae_max": args.ae_max,
        "grv_min": args.grv_min,
    }
    results = judge(stats, th)

    payload = {"stats": stats, "thresholds": th, "results": results}
    payload = attach_meta(payload, csv=args.csv, meta_path=args.meta)
    write_json(Path(args.json), payload)
    write_md(Path(args.md), stats, results, th, meta=payload.get("meta"))

    # Console summary
    print("== Audit Summary ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
