#!/usr/bin/env python3
"""Run text inference and compute ΔE internal metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from numpy.typing import NDArray

from core.metrics import calc_delta_e_internal


def _build_encoder(model_name: str) -> Callable[[str], NDArray[np.floating[Any]]]:
    """Return a text encoder. Uses transformers if available."""
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch

        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name)

        def encode(text: str) -> NDArray[np.floating[Any]]:
            tokens = tok(text, return_tensors="pt")
            with torch.no_grad():
                out = mdl(**tokens).last_hidden_state.mean(dim=1)
            return out.squeeze().numpy()

        return encode
    except Exception:
        def encode(text: str) -> NDArray[np.float_]:
            h = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
            return np.asarray([len(text), h], dtype=float)

        return encode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference on text input")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", type=str, help="prompt string")
    g.add_argument("--infile", type=Path, help="path to input text file")
    p.add_argument("--dump-hidden", type=Path, help="write hidden vectors to JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    encode = _build_encoder(args.model)

    if args.prompt is not None:
        lines = [args.prompt]
    else:
        content = args.infile.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]

    vecs: list[NDArray[np.floating[Any]]] = [encode(line) for line in lines]

    for i in range(1, len(vecs)):
        de = calc_delta_e_internal(vecs[i-1], vecs[i])
        print(f"{de:.3f}")

    if args.dump_hidden:
        with open(args.dump_hidden, "w", encoding="utf-8") as fh:
            for i, v in enumerate(vecs):
                json.dump({"turn": i, "hidden_state": v.tolist()}, fh)
                fh.write("\n")


if __name__ == "__main__":
    main()
