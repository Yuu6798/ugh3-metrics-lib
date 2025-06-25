import argparse
import warnings

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from bert_score import score as bert_score  # type: ignore[import-not-found]
from comet import load  # type: ignore[import-not-found, attr-defined]
from rouge_score import rouge_scorer
import fugashi

warnings.filterwarnings("ignore", category=FutureWarning)


def classify_domain(text: str) -> str:
    """Rudimentary domain classifier based on keywords."""
    t = text.lower()
    tech_kw = ["code", "algorithm", "python", "api", "技術", "テク", "アルゴリズム"]
    creative_kw = ["story", "fiction", "novel", "poem", "物語", "創作"]
    specialized_kw = ["医学", "法律", "finance", "経済", "化学", "物理"]
    if any(k in t for k in tech_kw):
        return "technical"
    if any(k in t for k in creative_kw):
        return "creative"
    if any(k in t for k in specialized_kw):
        return "specialized"
    return "general"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score QA pairs with baseline metrics")
    parser.add_argument("--input", required=True, help="input CSV path")
    parser.add_argument("--output", required=True, help="output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    candidates = df["answer"].astype(str).tolist()
    references = df["question"].astype(str).tolist()

    # BERTScore
    _, _, f1 = bert_score(candidates, references, lang="multilingual", rescale_with_baseline=True)
    # bert_score は torch.Tensor を返すので Python list に変換
    df["bertscore"] = f1.cpu().numpy().tolist()

    # ===== COMET-Kiwi（多言語BLEURT代替） =====
    comet_model = load("Unbabel/wmt22-cometkiwi-da")
    comet_scores = comet_model.predict(src=candidates, mt=candidates, ref=references)["scores"]
    df["cometkiwi"] = [float(s) for s in comet_scores]

    # ===== 日本語Rouge-L (MeCab+rouge_score) =====
    tagger = fugashi.Tagger()

    def tokenize_ja(text: str) -> str:
        return " ".join(tok.surface for tok in tagger(text))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_vals = [
        scorer.score(tokenize_ja(ref), tokenize_ja(pred))["rougeL"].fmeasure
        for pred, ref in zip(candidates, references)
    ]
    df["rougeL"] = rouge_vals

    df["domain"] = df["question"].apply(classify_domain)

    metrics: list[str] = ["por", "delta_e", "grv", "bertscore", "cometkiwi", "rougeL"]
    # 明示的に型注釈を付与して mypy の var-annotated エラーを回避
    adopt: np.ndarray[Any, Any] = np.ones(len(df), dtype=bool)
    for m in metrics:
        q_low, q_high = df[m].quantile([0.05, 0.95]).values
        adopt &= (df[m] >= q_low) & (df[m] <= q_high)
    df["adopted"] = adopt.astype(int)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
