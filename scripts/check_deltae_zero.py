#!/usr/bin/env python3
"""Flag rows where delta-e is zero and record the reason."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

REASON_ZERO_VECTOR = "ZERO_VECTOR"
REASON_IDENTICAL_TEXT = "IDENTICAL_TEXT"
REASON_UNKNOWN_CAUSE = "UNKNOWN_CAUSE"
REQUIRED_COLUMNS = {"question", "answer_a", "answer_b", "deltae"}


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".parq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV or Parquet matching the file extension."""
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def compute_reasons(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    """Compute delta-e zero flags and reasons for the DataFrame."""
    zero_mask = np.isclose(df["deltae"], 0.0)
    flags = pd.Series(False, index=df.index, dtype=bool)
    reasons = pd.Series([""] * len(df), index=df.index, dtype=object)

    zero_rows = df[zero_mask]
    if zero_rows.empty:
        df["deltae_zero_flag"] = flags
        df["deltae_zero_reason"] = reasons
        return df

    with torch.no_grad():
        q_emb = model.encode(zero_rows["question"].tolist(), convert_to_tensor=True)
        a_emb = model.encode(zero_rows["answer_a"].tolist(), convert_to_tensor=True)
        b_emb = model.encode(zero_rows["answer_b"].tolist(), convert_to_tensor=True)

    for i, idx in enumerate(zero_rows.index):
        q_vec = q_emb[i]
        a_vec = a_emb[i]
        b_vec = b_emb[i]
        q_norm = torch.linalg.norm(q_vec).item()
        a_norm = torch.linalg.norm(a_vec).item()
        b_norm = torch.linalg.norm(b_vec).item()
        qa_cos = torch.nn.functional.cosine_similarity(q_vec, a_vec, dim=0).item()
        ab_cos = torch.nn.functional.cosine_similarity(a_vec, b_vec, dim=0).item()
        text_identical = (
            df.at[idx, "question"] == df.at[idx, "answer_a"]
            or df.at[idx, "answer_a"] == df.at[idx, "answer_b"]
        )

        if q_norm < 1e-6 or a_norm < 1e-6 or b_norm < 1e-6:
            reason = REASON_ZERO_VECTOR
        elif text_identical or qa_cos > 0.999 or ab_cos > 0.999:
            reason = REASON_IDENTICAL_TEXT
        else:
            reason = REASON_UNKNOWN_CAUSE

        flags.at[idx] = True
        reasons.at[idx] = reason

    df["deltae_zero_flag"] = flags
    df["deltae_zero_reason"] = reasons
    return df


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check delta-e zero rows.")
    parser.add_argument("file", type=Path, help="CSV or Parquet file path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI."""
    args = parse_args(argv)
    path = args.file
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    try:
        df = load_dataframe(path)
    except Exception as e:
        print(f"Error reading file {path}: {e}", file=sys.stderr)
        return 1
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        print(f"Missing columns: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Embedding model load failed: {e}", file=sys.stderr)
        return 2

    df = compute_reasons(df, model)
    try:
        save_dataframe(df, path)
    except Exception as e:
        print(f"Error saving file {path}: {e}", file=sys.stderr)
        return 1

    print(f"Wrote {len(df)} rows to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
