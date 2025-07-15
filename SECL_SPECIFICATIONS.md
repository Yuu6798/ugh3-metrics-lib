# SECL（Self-Evolving Cognitive Learning）技術仕様書

## 1. システム概要

SECLプログラムは、進化型認知学習アルゴリズムを用いてQ&Aサイクルをシミュレーションします。主な目的は、自己進化を通じて知識を拡張し、新しい洞察を得ることです。

## 2. データ構造

### HistoryEntryクラス

HistoryEntryクラスは、過去のQ&Aサイクルの履歴を保持します。各フィールドの詳細は以下の通りです：

- `question`: 提出された質問
- `answer_a`: ひとつ前の回答
- `answer_b`: 現在の回答
- `por`: PoR スコア
- `delta_e`: ΔE スコア
- `grv`: GRV スコア
- `score`: 総合スコア
- `spike`: スパイク判定フラグ
- `external`: 外部問い合わせフラグ
- `anomaly_por`: PoR 異常判定
- `anomaly_delta_e`: ΔE 異常判定
- `anomaly_grv`: GRV 異常判定
- `por_null`: PoR 計算不能フラグ
- `score_threshold`: スコア閾値
- `timestamp`: エントリの作成日時

備考: 旧フィールド `answer` は後方互換のため `@property` で参照できます。

## 3. コアアルゴリズム

### novelty_score（新奇性計算）

novelty_scoreは、質問と回答の新規性を評価するための指標です。

### simulate_delta_e（エネルギー変化）

simulate_delta_eは、システムのエネルギー状態の変化をシミュレートします。

### calc_grv_field（語彙重力場）

calc_grv_fieldは、語彙間の重力場を計算し、関連性を評価します。

### 停滞検知と創発メカニズム

システムの停滞を検知し、新たな創発を促進するメカニズムを備えています。

## 4. 設定パラメータ

config.jsonファイルには、システムの動作を制御するためのパラメータが含まれています。各パラメータの影響と推奨値は以下の通りです：

- `learning_rate`: 学習率（推奨値: 0.01）
- `threshold`: 新奇性スコアの閾値（推奨値: 0.5）

## 5. 外部連携機能

### CSV/JSON出力仕様

システムは、Q&Aサイクルの結果をCSVまたはJSON形式で出力できます。

### HTTP Alert機能

特定の条件が満たされた場合、HTTPリクエストを通じてアラートを送信します。

### 自動バックアップ仕様

定期的にデータをバックアップし、システムの信頼性を向上させます。
