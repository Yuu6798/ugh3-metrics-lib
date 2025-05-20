# ugh3-metrics-lib

UGHer PoR・ΔE OSSライブラリ＋リファレンス実装 (UGHer PoR/ΔE open source library and reference implementation).
本リポジトリはPoR (Proof of Resonance) やΔE (存在エネルギー差)、grv (語彙重力) といった指標をPythonから簡単に計算するためのツール集です。

## 概要 / Overview
UGHer理論に基づき、AI内在ダイナミクスを評価するための基本的な数値指標を提供します。シンプルな実装なので、研究用途や他プロジェクトへの組み込みの参考実装として利用できます。

## Features / 特徴
- PoR（共鳴点）トリガー計算
- ΔE（存在エネルギー差）スコア計算
- grv（語彙重力）メトリクス計算
- 最小限の依存関係でシンプル
- Python 3.8+対応

## Installation / インストール
```bash
pip install git+https://github.com/Yuu6798/ugh3-metrics-lib.git
# または
git clone https://github.com/Yuu6798/ugh3-metrics-lib.git
```

## Quick Start / クイックスタート
```python
from por_trigger import por_trigger
from deltae_scoring import deltae_score
from grv_scoring import grv_score

result = por_trigger(q=1.0, s=0.9, t=0.8, phi_C=1.05, D=0.1)
print(result)
print(deltae_score(E1=10.0, E2=12.5))
print(grv_score("AIは問いに答える存在です"))
```

## Usage / 使い方
`por_trigger` はPoRイベント発生可否を示す辞書を返します。`deltae_score` はエネルギー差、`grv_score` は語彙の広がりを数値化します。詳細なAPI仕様は各モジュールのドキュメントを参照してください。

<!-- AUTO SECTION START -->
### Import Examples / インポート例
```python
from por_trigger import por_trigger
from deltae_scoring import deltae_score
from grv_scoring import grv_score
```

### Function Overview / 関数概要
- `por_trigger(q, s, t, phi_C, D, *, theta=0.6) -> dict`
- `deltae_score(E1, E2) -> float`
- `grv_score(text, *, vocab_limit=30) -> float`

<!-- AUTO SECTION END -->

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

<!-- TODO: 3指標の関係を図示したイメージを追加する -->

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
- `por_trigger.py` – PoRトリガー計算
- `deltae_scoring.py` – ΔEスコア計算
- `grv_scoring.py` – grvメトリクス計算
- `secl_qa_cycle.py` – メトリクスを使用したQ&Aサイクル例
- `por_deltae_grv_collector.py` – 簡易データ収集CLI
- `design_sketch.py` – コア計算のリファレンス実装
- `tests/` – ユニットテスト

## Contribution / コントリビュート
改善提案やバグ報告はPull RequestまたはIssueでお願いします。

## License / ライセンス
MIT License

## Contact / 連絡先
ご質問や要望はGitHub Issueよりご連絡ください。
