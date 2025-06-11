import argparse
import warnings

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from bert_score import score as bert_score  # type: ignore[import-not-found]
import evaluate  # type: ignore[import-not-found]

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

    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=candidates, references=references)
    df["rougeL"] = rouge_res["rougeL"]

    # BLEURT: 日本語でモデルが無い場合は NaN を埋める
    try:
        bleurt = evaluate.load("bleurt")
        bleurt_res = bleurt.compute(predictions=candidates, references=references)
        df["bleurt"] = [float(s) for s in bleurt_res["scores"]]
    except Exception:
        df["bleurt"] = np.nan

    df["domain"] = df["question"].apply(classify_domain)

    metrics = ["por", "delta_e", "grv", "bertscore", "bleurt", "rougeL"]
    adopt = np.ones(len(df), dtype=bool)
    for m in metrics:
        q_low, q_high = df[m].quantile([0.05, 0.95]).values
        adopt &= (df[m] >= q_low) & (df[m] <= q_high)
    df["adopted"] = adopt.astype(int)

    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
