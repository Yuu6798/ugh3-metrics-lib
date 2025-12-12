![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Dataset](https://github.com/Yuu6798/ugh3-metrics-lib/actions/workflows/build_dataset.yml/badge.svg)

# ugh3-metrics-lib

UGHer PoR・ΔE・grv のリファレンス実装 (UGHer PoR/ΔE/grv open source library and reference implementation).
本リポジトリは PoR (Point of Resonance)、ΔE (Delta E)、grv (Gravity of Lexicon) を Python から計算するための公式実装と
ドキュメントを提供します。

## 概要 / Overview
UGHer/UGH3 理論に基づき、AI 内在ダイナミクスを評価するための基本的な数値指標を提供します。シンプルな実装なので、研究用途や他プロジェクトへの組み込みの参考実装として利用できます。

## Features / 特徴
- PoR（共鳴点）/ ΔE / grv のコア 3 指標を提供（PoR と grv は 0–1 正規化、ΔE は Mode A をデフォルトで 0–1 で返却）
- PorV4 / DeltaE4 / GrvV4 など論文・既存実装に対応した API を同梱
- しきい値（θ_fire）と評価用の τ_eval を分離したポリシー設計
- 最小限の依存関係でシンプル
- Python 3.8+対応

## Installation / インストール

### Users
```bash
pip install por-deltae-lib
```
The first ΔE or PoR call that loads sentence-transformers will download
`all-MiniLM-L6-v2` into `~/.cache/torch`. To prefetch the model ahead of time:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
PY
```

### Development / CI
```bash
pip install -e .[dev]
```
The `[dev]` extras group installs all tools needed for testing and
type-checking, including `pytest`, `mypy`, and other optional packages.

> **Migration note:** v0.1.0 から依存は `pyproject.toml` に一本化されました。
> 開発者は `pip install -e .[dev]` を実行してください。

## Quick Start / クイックスタート
リファレンス実装のクラスは `ugh3_metrics.metrics` から直接 import できます。
PoR と grv のデフォルト実装は **0.0〜1.0** で返却します。ΔE は Mode A で 0–1 のテキスト代理スコアを返し、Mode B では内部 drift をそのまま返す実験パスを想定しています。

```python
from ugh3_metrics.metrics import PorV4, DeltaE4, GrvV4
from core.metrics import POR_FIRE_THRESHOLD

por = PorV4(auto_load=True).score("Q: How to align?", "Prior resonance topic")
deltae = DeltaE4().score("prev answer", "current answer")
grv = GrvV4().score("AIは問いに答える存在です", "")

print({
    "por": por,
    "por_fire": por >= POR_FIRE_THRESHOLD,  # θ_fire default policy (0.82)
    "deltae": deltae,
    "grv": grv,
})
```

### CLI (論文準拠の最短実行例)
コマンドラインから PoR/ΔE/grv を一括収集する場合は、論文と同じ `facade/collector.py` を呼び出せます。

```bash
python facade/collector.py \
  --input examples/prompts.csv \
  --output out.json \
  --por \
  --delta_e \
  --grv
```

### CLI (運用例・自動収集)
論文掲載例とは別に、`--auto` を使った自動収集の運用例も利用できます。

```bash
python facade/collector.py --auto -n 10 --q-provider openai --ai-provider openai \
  --quiet --summary -o runs/demo_por_history.csv
```

## Metric definitions / コア定義（canonical）
### 1. PoR（Point of Resonance）
- **定義：** 照合強度。問い合わせ $Q$ と応答候補集合 $\{R_i\}$ の最大類似度。
- **数式：**
  \[
    \text{por\_raw}(Q,\{R_i\}) = \max_i \text{sim}(Q, R_i)
  \]
- 図表などで扱う連続値の **por_raw** と、イベント判定用の **por_fire**（θ_fire 超え）を常に分けて解釈してください。
- **発火判定：**
  - `por_fire = (por_raw >= θ_fire)` 。ライブラリ既定の θ_fire は `POR_FIRE_THRESHOLD` (= 0.82) です。
  - **評価用の閾値 τ_eval は別に設定**し、ポリシー（発火イベント）と分類評価を混同しないようにしてください。

### 2. ΔE（Delta E）
互換性のある 2 つのモードを提供します（Mode A は 0–1、Mode B は raw drift を返す運用も想定）。

- **Mode A（テキスト代理; デフォルト）**
  - \( \Delta E = 1 - \text{sim}(R_{\text{prev}}, R_{\text{curr}}) \)
  - `DeltaE4.score(a, b)` がこれに対応します（cosine 類似度が負になるケースも含めて [0,1] に正規化）。
- **Mode B（内部状態; オプション）**
  - トピック KL / attention / logits など任意の内部 drift をそのまま取り扱うモードです。raw drift 値（上限なし）を返却し、使用した `deltae_method` を必ずログに残すポリシーを推奨します。
  - 必要に応じて正規化例を適用できます（例: `deltae_norm = 1 - exp(-kl / k)` または `clip(kl / k, 0, 1)` など、k はドメイン依存）。
  - 行列入力がある場合は `core.metrics.calc_delta_e_internal` を利用して 1 − cosine を計算できます。

> 旧 API にある `deltae_score(E1, E2)` のような単なる実数差分は「ΔEではない別指標（例: energy_delta）」として扱い、分類評価には使用しないでください。

### 3. grv（Gravity of Lexicon）
- **定義：** TF‑IDF 集中度、ローカルエントロピー、共起密度を [0,1] に正規化し、重み付き合成した語彙重力。
- **数式イメージ：**
  \[
    \text{grv} = w_1 \cdot \text{tfidf\_topk} + w_2 \cdot \text{local entropy} + w_3 \cdot \text{cooccurrence density}
  \]
- `GrvV4.score(text, "")` がこの合成を実装しています。抽象度/難解度/本質度などのラベルは解釈補助として別枠で扱い、コア定義には含めません。

## Configuration / しきい値と設定
- `core.metrics.POR_FIRE_THRESHOLD = 0.82` : PoR 発火用 θ_fire（ポリシー閾値）。
- 評価用の τ_eval はドメインに応じて別途設定してください（例: 0.6 など）。
- `HIGH_DELTA_TH` : SECL 状態判定用 ΔE 上限の既定値 0.85（`config/config.toml` などで上書き可能）。
その他の設定は `config/` ディレクトリの YAML/TOML を参照してください。

## Domain profiles / ドメイン別の考え方
テキスト / 画像 / マルチモーダルで閾値は普遍ではありません。`theta_fire` や τ_eval はデータ分布に応じて再調整し、ドメイン毎にしきい値ログを残してください。

## Examples / 実行例
実際の計算フローは `examples/ugh3_metrics_demo.ipynb` を参照してください。派生概念の PoR_chain / PoR_wave / PoR_convergence は実験ログ由来の上位概念として整理されており、コア API の出力を組み合わせて利用します。

## Reference Metrics / リファレンス実装
PorV4, DeltaE4, GrvV4, and SciV4 are provided under `ugh3_metrics.metrics`. Integration tests in `tests/` cover these modules; `tests/test_deltae_v4_setparams.py` verifies `DeltaE4` using a dummy embedder.

<!-- AUTO SECTION START -->
### Import Examples / インポート例
```python
from ugh3_metrics.metrics import PorV4, DeltaE4, GrvV4
from core.metrics import POR_FIRE_THRESHOLD, calc_delta_e_internal
```

### Function Overview / 関数概要
- `PorV4(auto_load=True).score(query, reference) -> float`
- `DeltaE4(embedder=None).score(prev_text, curr_text) -> float`
- `GrvV4().score(text, unused="") -> float`
- `calc_delta_e_internal(prev_vec, new_vec) -> float`  # 内部状態用 [0,1]

<!-- AUTO SECTION END -->

### Visualization Utilities / 可視化用ユーティリティ

`phase_map_heatmap.py` ではメトリクス履歴をヒートマップとして描画します。
`visualize_tensor.py` はデモ用テンソルを生成して可視化します。

```bash
python examples/phase_map_heatmap.py
python examples/visualize_tensor.py --demo
```

生成結果の一例を `images/phase_map.png` と `images/tensor.png` に掲載しています。

![Phase Map](images/phase_map.png)
![Tensor](images/tensor.png)

### Dataset / Reproducibility
- 現行の curated データセットは `data/por_history_20250523_0043.csv`（CSV 形式、20 件）に保存されています。論文と同様に、今後 ~300 件への拡張を予定しています。
- `scripts/build_dataset.py` で CSV/Parquet を再構成、`scripts/recalc_scores_v4.py` で PoR/ΔE/grv を再計算できます。
- ナイトリーワークフローが `datasets/` 以下に生成物を出力する設定もありますが、RAW_DATA_URL や Secrets が揃っている場合に限られます（常時自動更新を保証するものではありません）。
- `scripts/detect_duplicates.py` や `scripts/audit_workflows.py` を用いて重複検出やワークフロー監査も可能です。

```bash
python scripts/build_dataset.py --out-parquet datasets/current_recalc.parquet --out-csv datasets/current.csv
python scripts/recalc_scores_v4.py --infile runs/deltae_log.csv --outfile runs/metrics_recalc.parquet
```

### Workflow failure on embedding errors
`DeltaE4` raises `RuntimeError` when no embedding model is available or a zero-vector is returned.  \
Workflows invoking `recalc_scores_v4.py` therefore exit with status 1 in such cases.

### build_dataset dual output
Generate both CSV and Parquet in one run:
```bash
python scripts/build_dataset.py --out-csv datasets/current.csv --out-parquet datasets/current.parquet
```

### Duplicate check
```bash
python scripts/detect_duplicates.py --outdir dup_report
open dup_report/report_dup.md
```

### Workflow audit
```bash
python scripts/audit_workflows.py --out report/workflow_audit.md
open report/workflow_audit.md
```

## Safety / Ethical note
高 PoR は semantic mimicry でも達成でき、真実性・倫理性を保証するものではありません。ログ保存や可視化（例: phase map heatmap）による監査を推奨します。

## PoR Automation / 日次自動化パイプライン

| Workflow | Cron (UTC) | JST | 主要ステップ | 成果物 |
|-----------|-----------|-----|-------------|--------|
| **Nightly Collect & Build Dataset** | `30 15 * * *` | 00:30 | Prefetch model → Auto collect QA logs → Filter ΔE==0 → Upload dataset | `runs/${DATE}/cycle.csv`, `datasets/${DATE}/dataset.{parquet,csv}` (Secrets が揃う場合) |

1. `nightly-collect-build-dataset.yml` が自動で Q&A を収集し PoR / ΔE / grv を計算、ΔE==0 を除去してデータセットを生成します。
2. Secrets に **OPENAI_API_KEY** と **PAT_TOKEN** を設定するだけで、日次パイプラインが稼働します。

パイプラインを有効化している場合のみ `./datasets/` 以下に日次成果物が追加されます。手動配布の curated データは `data/por_history_20250523_0043.csv` を参照してください。

```bash
# 手動テスト (GitHub Actions の Run workflow でも可)
python facade/collector.py --auto -n 10 --q-provider openai --ai-provider openai \
  --quiet --summary --output test_por.csv
# ドメインと難易度は内部で自動的にばらつきを付けて生成されます
```

## Contribution / コントリビュート
改善提案やバグ報告はPull RequestまたはIssueでお願いします。

## Maintainers
This project is primarily maintained by [Yuu6798](https://github.com/Yuu6798).
Maintainers should set the `ST_CACHE` environment variable to a persistent
cache directory to avoid re-downloading models during CI runs.

## License / ライセンス
Code: MIT License
Dataset: CC-BY 4.0 License

## Contact / 連絡先
ご質問や要望はGitHub Issueよりご連絡ください。
