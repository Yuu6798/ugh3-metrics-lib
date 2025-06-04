以下に整理されたフォルダツリー構造と各ファイルの役割を示します。
│   ├── unified-ai-issue-to-pr.yml  # 統合AIワークフロー（Issue→AI→PR完全自動化）
│   ├── ci.yml                      # CI/テスト実行ワークフロー
│   ├── typecheck.yml               # 型チェック専用ワークフロー
│   └── secret-smoke.yml            # シークレット検証ワークフロー
# イシュー連動コード生成自動化プログラム仕様

## 1. プログラム概要
自然言語のイシュー本文から自動でコード生成・プルリクエスト作成を行うプログラムです。リアルタイムプログレス表示機能を備え、CI/CD統合による品質保証を実現します。

## 2. システム構成
```
📁 システム構成
各ファイルの役割を明確にし、構成図と機能説明を適切に分離しました。
重複記述を完全に削除し、クリーンな構成を実現しました。
├── .github/workflows/
│   ├── ci.yml                   # CI/テスト実行ワークフロー
unified-ai-issue-to-pr.yml: 統合処理（Issue検証→AI解析→コード生成→PR作成→進捗管理）
- 競合排除による安定実行
- 条件分岐による適応的処理
- peter-evans/create-pull-request使用
- 統一されたProgress tracker管理
- 競合排除による安定実行
- 条件分岐による適応的処理
- peter-evans/create-pull-request使用
- 統一されたProgress tracker管理
│   ├── typecheck.yml            # 型チェック専用ワークフロー
│   └── secret-smoke.yml         # シークレット検証ワークフロー
└── scripts/
    ├── ai_issue_codegen.py      # AIコード生成エンジン（メイン処理）
    ├── progress_tracker.py      # リアルタイムプログレス表示システム
    └── recalc_deltae.py         # ΔE再計算ユーティリティ
```

## 3. 動作フロー
CI/CD: GitHub Actions (4つのワークフロー連携)
CI/CD: GitHub Actions (4つのワークフロー連携) - 統合により効率化・安定性向上
1. **Issue作成** → GitHub Actionsトリガー
2. **プログレス初期化** → Issue内にリアルタイム進捗表示
3. **AI解析** → OpenAI APIでイシュー本文解析
4. **コード生成** → LLMによる具体的な修正内容生成
5. **ファイル適用** → 生成されたdiffを実際のファイルに適用
6. **PR作成** → PAT_TOKENでプルリクエスト自動作成
7. **CI実行** → 自動テスト・品質チェック実行

Issue検証フェーズ: 空のIssue・キーワード不一致の早期検出
条件実行: Issue内容に応じた処理選択（AI実行/スキップ）
エラーハンドリング強化: continue-on-error による安定性向上
統一認証: PAT_TOKEN による一貫した権限管理
Progress統一管理: 重複排除による正確な進捗表示
## 4. 技術仕様
- **言語**: Python 3.8+
- **AI**: OpenAI API (GPT-4推奨)
- **CI/CD**: GitHub Actions (5つのワークフロー連携)
- **認証**: PAT_TOKEN (Personal Access Token)
- **対応形式**: Unified Diff形式での変更適用
- **型チェック**: MyPy統合
競合排除設計: 重複実行防止による安定性向上
統合ワークフロー: 単一責任による保守性改善
条件分岐処理: Issue内容に応じた適応的実行
エラーハンドリング: 各段階での適切な例外処理
統一進捗管理: 一貫したProgress tracker運用
- **品質保証**: 複数段階のCI/CDパイプライン

## 5. 設定要件
- `OPENAI_API_KEY`: OpenAI APIキー
- `PAT_TOKEN`: GitHub Personal Access Token (repo権限)
- `AI_MODEL`: 使用するAIモデル (デフォルト: gpt-4)

統合による改善: ワークフロー競合問題の完全解決
安定性向上: Git操作競合エラーの排除
処理効率化: 重複実行による無駄なリソース消費の削減
## 6. ワークフロー連携
- **issue-to-pr.yml**: メイン処理（Issue解析→AI生成→PR作成）
- **ci.yml**: PR作成後の自動テスト実行
- **typecheck.yml**: 型安全性の継続的チェック
- **secret-smoke.yml**: セキュリティ検証の自動実行
- **ai-issue-codegen.yml**: AI専用処理ワークフロー

## 7. 品質保証
- PAT_TOKEN使用によるCI自動実行
- テスト結果のプルリクエスト表示
- 自動品質チェック機能

## 8. アーキテクチャ特徴
- マイクロサービス型ワークフロー設計
- 責任分離による高い保守性
- 型安全性重視の開発手法
- セキュリティファーストなアプローチ
- 段階的拡張が可能な柔軟性

## 9. 制限事項・注意点
- OpenAI API使用料金
- GitHub Actions実行時間制限
- PAT_TOKEN権限設定の重要性
- LLM生成内容の品質依存性