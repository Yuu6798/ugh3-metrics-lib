
## 1. プログラム概要
自然言語のイシュー本文から自動でコード生成・プルリクエスト作成を行うプログラムです。リアルタイムプログレス表示機能を備え、CI/CD統合による品質保証を実現します。

## 2. システム構成
```
📁 システム構成
├── .github/workflows/
│   ├── unified-ai-issue-to-pr.yml  # 統合AIワークフロー（Issue→AI→PR完全自動化）
│   ├── ci.yml                      # CI/テスト実行ワークフロー
│   ├── typecheck.yml               # 型チェック専用ワークフロー
│   └── secret-smoke.yml            # シークレット検証ワークフロー
└── scripts/
    ├── ai_issue_codegen.py         # AIコード生成エンジン（メイン処理）
    ├── progress_tracker.py         # リアルタイムプログレス表示システム
    └── recalc_deltae.py            # ΔE再計算ユーティリティ
```

## 3. 動作フロー
Issue作成 → GitHub Actionsトリガー
プログレス初期化 → Issue内にリアルタイム進捗表示
AI解析 → OpenAI APIでイシュー本文解析
コード生成 → LLMによる具体的な修正内容生成
ファイル適用 → 生成されたdiffを実際のファイルに適用
PR作成 → PAT_TOKENでプルリクエスト自動作成
CI実行 → 自動テスト・品質チェック実行

## 4. 技術仕様
- 言語: Python 3.8+
- AI: OpenAI API (GPT-4推奨)
- CI/CD: GitHub Actions (4つのワークフロー連携)
- 認証: PAT_TOKEN (Personal Access Token)
- 対応形式: Unified Diff形式での変更適用
- 型チェック: MyPy統合
- 品質保証: 複数段階のCI/CDパイプライン

## 5. 設定要件
- OPENAI_API_KEY: OpenAI APIキー
- PAT_TOKEN: GitHub Personal Access Token (repo権限)
- AI_MODEL: 使用するAIモデル (デフォルト: gpt-4)

## 6. ワークフロー連携
- unified-ai-issue-to-pr.yml: 統合処理（Issue解析→AI生成→PR作成）
- ci.yml: PR作成後の自動テスト実行
- typecheck.yml: 型安全性の継続的チェック
- secret-smoke.yml: セキュリティ検証の自動実行

## 7. 品質保証
- PAT_TOKEN使用によるCI自動実行
- テスト結果のプルリクエスト表示
- 自動品質チェック機能

## 8. アーキテクチャ特徴
- 統合ワークフロー設計による安定性向上
- 責任分離による高い保守性
- 型安全性重視の開発手法
- セキュリティファーストなアプローチ
- 段階的拡張が可能な柔軟性

## 9. 制限事項・注意点
- OpenAI API使用料金
- GitHub Actions実行時間制限
- PAT_TOKEN権限設定の重要性
- LLM生成内容の品質依存性
