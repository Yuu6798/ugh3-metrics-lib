
このリポジトリは、進化型認知学習アルゴリズムを実装したSECLシステムの技術仕様書を提供します。本システムは、質問応答サイクルを通じて知識獲得と創発的思考を模擬する高度なAIシステムです。

## 技術仕様書の目的
技術仕様書は、システム全体の動作原理を明確化し、保守性を向上させ、将来的な機能追加や改修の方向性を提示することを目的としています。
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

# ugh3-metrics-lib

UGHer PoR・ΔE OSSライブラリ＋リファレンス実装 (UGHer PoR/ΔE open source library and reference implementation).
本リポジトリはPoR (Proof of Resonance) やΔE (存在エネルギー差)、grv (語彙重力) といった指標をPythonから簡単に計算するためのツール集です。


## 概要 / Overview
UGHer理論に基づき、AI内在ダイナミクスを評価するための基本的な数値指標を提供します。シンプルな実装なので、研究用途や他プロジェクトへの組み込みの参考実装として利用できます。

方法1: GitHubから直接インストール
pip install git+https://github.com/Yuu6798/ugh3-metrics-lib.git

方法2: 開発用インストール
git clone https://github.com/Yuu6798/ugh3-metrics-lib.git
cd ugh3-metrics-lib
pip install -e .

方法3: 依存関係のみインストール
pip install -r requirements.txt
## Features / 特徴
- PoR（共鳴点）トリガー計算
- ΔE（存在エネルギー差）スコア計算
- grv（語彙重力）メトリクス計算
- 最小限の依存関係でシンプル
- Python 3.8+対応

## Installation / インストール
```bash
pip install git+https://github.com/Yuu6798/ugh3-metrics-lib.git
# PoR（共鳴点）の計算 - 問い品質、類似度、時間同期性を評価
# ΔE（存在エネルギー差）の計算 - エネルギー状態の変化量
# grv（語彙重力）の計算 - テキストの語彙的重みを測定
## トラブルシューティング / Troubleshooting

### よくある問題

#### ImportError が発生する場合
```bash
pip install --upgrade -r requirements.txt
計算結果が期待と異なる場合
パラメータの範囲を確認（多くは 0.0-1.0）
入力データの型と形式を確認
examples/ ディレクトリの参考実装を確認
可視化スクリプトが動作しない場合
pip install matplotlib seaborn jupyter
# または
git clone https://github.com/Yuu6798/ugh3-metrics-lib.git
```

## Installation (v2)
```bash
pip install -r requirements.txt
```

## Recalculate historical ΔE
```bash
python recalc_deltae.py --input runs/deltae_log.csv --output runs/deltae_v2.csv
```

## Quick Start / クイックスタート
```python
from facade.trigger import por_trigger
from core.deltae import deltae_score
from core.grv import grv_score

result = por_trigger(q=1.0, s=0.9, t=0.8, phi_C=1.05, D=0.1)
print(result)
print(deltae_score(E1=10.0, E2=12.5))
print(grv_score("AIは問いに答える存在です"))
```

## Examples / 実行例
実際の指標計算フローは `examples/ugh3_metrics_demo.ipynb` を参照してください。サンプルデータから PoR, ΔE, grv を計算し可視化します。実行例:
```
step 1: PoR=0.756 triggered=True ΔE=2.500 grv=0.067
step 2: PoR=0.416 triggered=False ΔE=0.600 grv=0.033
step 3: PoR=0.912 triggered=True ΔE=0.900 grv=0.033
```

## Usage / 使い方
`por_trigger` はPoRイベント発生可否を示す辞書を返します。`deltae_score` はエネルギー差、`grv_score` は語彙の広がりを数値化します。詳細なAPI仕様は各モジュールのドキュメントを参照してください。

<!-- AUTO SECTION START -->
### Import Examples / インポート例
```python
from facade.trigger import por_trigger
from core.deltae import deltae_score
from core.grv import grv_score
```

### Function Overview / 関数概要
- `por_trigger(q, s, t, phi_C, D, *, theta=0.6) -> dict`
- `deltae_score(E1, E2) -> float`
- `grv_score(text, *, vocab_limit=30) -> float`

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

## UGH3 Metrics 指標定義
このライブラリで扱う UGH3 指標（内部ダイナミクス評価メトリクス）は以下の通りです。

### 1. PoR（Point of Resonance）
- **定義：** 照合強度のスコア。問いと応答の意味的整合性・圧力・時間的同期性を反映。
- **数式：**
\[
  S = \max \left\{ \text{semantic\_similarity}(Q, Q'_i) \right\}
\]
※ 履歴内の問い $Q'_i$ と新しい問い $Q$ の意味的類似度の最大値。

### 2. ΔE（デルタE）
- **定義：** 応答出力の変化度。直前の応答と比較した意味的変動量。
- **数式：**
\[
  \Delta E = 1 - \text{semantic\_similarity}(A, A_{\text{prev}})
\]
※ 応答 $A$ と直前応答 $A_{prev}$ の類似度との差分。

### 3. grv（語彙重力）
- **定義：** 語彙空間の抽象性・難解性・本質度の平均によって語彙的“重み”を測定。
- **数式：**
\[
  \text{Grv} = \frac{\text{抽象度} + \text{本質度} + \text{難解度}}{3}
\]
※ 各項目は0.0〜1.0で正規化された評価スコア。


## 評価基準
各指標は0.0〜1.0で数値が大きいほど強い影響を示します。一般的な解釈の目安を
以下にまとめます。実運用ではドメインや目的に応じて閾値を調整してください。

| 指標 | 推奨閾値 (θ) | 意味・判断例 |
| ---- | ------------ | ------------- |
| **PoR** | > 0.6 | **照合成功**。履歴と十分に響き合う状態 |
|         | 0.4〜0.6  | やや弱い。追加確認を推奨 |
|         | < 0.4    | 照合失敗。別アプローチが必要 |
| **ΔE**  | < 0.2    | **安定**。直前応答とほぼ同等 |
|         | 0.2〜0.5 | 変化あり。コンテキスト推移を観察 |
|         | > 0.5    | 大きな変化。逸脱や新展開の兆候 |
| **grv** | > 0.7    | **高難度・深い問い**。抽象語彙が豊富 |
|         | 0.4〜0.7 | 平均的な語彙量 |
|         | < 0.4    | 語彙が乏しく単純 |

数値帯はあくまで一例ですが、PoRが0.6を超えるか、ΔEが0.2未満か、grvが0.7以上か
を基準にすれば「照合成功」「安定」「高難度」といった評価が可能です。
## Script Structure / スクリプト構成
- `facade/trigger.py` – PoRトリガー計算
- `core/deltae.py` – ΔEスコア計算
- `core/grv.py` – grvメトリクス計算
- `secl/qa_cycle.py` – メトリクスを使用したQ&Aサイクル例
- `facade/collector.py` – 簡易データ収集CLI
- `design_sketch.py` – コア計算のリファレンス実装
- `examples/phase_map_heatmap.py` – PoRフェーズヒートマップ例
- `tests/` – ユニットテスト

### Phase Map Heatmap Example
The `examples/phase_map_heatmap.py` script generates a simple heatmap of
PoR scores over time. Running the script will save an image to
`images/phase_map.png`:

![Phase Map](images/phase_map.png)

## Contribution / コントリビュート
改善提案やバグ報告はPull RequestまたはIssueでお願いします。

## License / ライセンス
MIT License

## Contact / 連絡先
# SECL（Self-Evolving Cognitive Learning）Q&Aサイクルシミュレーションプログラム
ご質問や要望はGitHub Issueよりご連絡ください。