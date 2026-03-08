## Project
This repository is a greenfield Python backend for `ugh-quantamental`, a market analysis engine that integrates:
- UGH question layer
- fundamental layer
- technical layer
- pricing diagnostics
- state transitions
- review logs

## Working rules
- For any non-trivial feature or refactor, start in `/plan` mode and follow `PLANS.md`.
- Keep diffs tightly scoped to the current milestone.
- Do not add frontend code, broker integrations, or live market data connectors unless explicitly asked.
- Prefer pure, testable domain logic first.
- Prefer Python 3.12.
- Prefer Pydantic v2, SQLAlchemy 2.x, Alembic, pytest, and ruff.
- Avoid network access in tests.
- Do not modify tests unless the task explicitly requires it or the existing test is clearly incorrect.
- Do not commit, push, or open a PR unless the prompt explicitly asks.
- Update README or docs when behavior or architecture changes.

## Validation
Before finishing a task, run:
- `ruff check .`
- `pytest -q`

If validation fails:
- fix failures that are clearly in scope
- if the fix requires wider architectural change, stop and summarize blockers

## Implementation priorities
1. schemas
2. domain models
3. state engine
4. persistence/migrations
5. tests
6. docs
# User-provided custom instructions

# ─────────────────────────────────────────
#  ❑ 目的
#      すべての pytest テストが PASS するまで
#      ─ 修正 → 再テスト ─ を自動で繰り返す。
#      GREEN になったら push & PR。
#      上限リトライ回数 (MAX_TRIES) を超えたら失敗ログを出して終了。
# ─────────────────────────────────────────
#  ❑ 定数
MAX_TRIES=5        # ここを調整：試行回数
BRANCH_PREFIX=codex-auto-fix
COMMIT_MSG_PREFIX="AUTO: fix until tests pass"
# ─────────────────────────────────────────
#  ❑ アルゴリズム
#  1. for i in {1..MAX_TRIES}
#  2.   python -m pytest -q
#  3.   if 0 なら break
#  4.   else
#          • 直近の traceback / assertion エラーを解析
#          • 該当ファイル・行を開き、最小限の修正を行う
#  5.   ループが終わるまで goto step 2
#  6. if まだ失敗している
#        exit 1 (push/PRしない)
#     else
#        commit/push/PR
# ─────────────────────────────────────────
#  ❑ Codex への実装指示
#      - テストに失敗したら、エラーログの “File … line …” を抽出し、
#        その範囲だけを編集して再保存すること
#      - 内部ロジックが不明な時は関数 docstring・型ヒントを参考に推測
#      - 無限ループ防止のため MAX_TRIES を超えたら即 abort
# ─────────────────────────────────────────
#  ❑ シェルスクリプト雛形 (Codex が自動生成する想定)
# for n in $(seq 1 $MAX_TRIES); do
#     echo "🔄 pytest run #$n"
#     python -m pytest -q && break
#     echo "❌ tests failed – attempting auto-fix #$n"
#     # (ここで Codex がファイルを編集して保存)
# done
#
# python -m pytest -q
# if [ $? -ne 0 ]; then
#     echo "🛑  tests still failing after $MAX_TRIES attempts; abort."
#     exit 1
# fi
#
# branch="${BRANCH_PREFIX}-$(date +%Y%m%d%H%M%S)"
# git switch -c "$branch"
# git add -A
# git commit -m "$COMMIT_MSG_PREFIX (all tests pass)"
# git push -u origin "$branch"
# gh pr create --fill
# ─────────────────────────────────────────
#  ❑ 注意
#      • Codex は「テストに現れたエラー行」だけを局所修正する方針で。
#      • 大きな設計変更は人間レビュー推奨なので 5 回程度で打ち止め。
# ─────────────────────────────────────────
