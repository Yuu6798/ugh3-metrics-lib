#!/usr/bin/env python3
"""
GitHub Issue Progress Tracker
Issue作成後にプログレスコメントを投稿し、各ステップで動的更新する
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

import urllib.error
import urllib.parse
import urllib.request


class ProgressTracker:
    def __init__(self, issue_number: int) -> None:
        self.issue_number: int = issue_number
        self.comment_id: Optional[int] = None
        self.tasks: List[Dict[str, str]] = []
        self.repo: Optional[str] = os.getenv("GITHUB_REPOSITORY")
        self.token: Optional[str] = os.getenv("GITHUB_TOKEN")

        if not self.repo or not self.token:
            raise ValueError("GITHUB_REPOSITORY and GITHUB_TOKEN must be set")

    def add_task(self, description: str, emoji: str = "⏳") -> None:
        """タスクをプログレスリストに追加"""
        timestamp: str = datetime.now().strftime("%H:%M:%S")
        self.tasks.append(
            {
                "description": description,
                "emoji": emoji,
                "status": "pending",
                "timestamp": timestamp,
            }
        )

    def complete_task(self, index: int, success: bool = True) -> None:
        """タスクを完了状態に更新"""
        if 0 <= index < len(self.tasks):
            self.tasks[index]["status"] = "completed" if success else "failed"
            self.tasks[index]["emoji"] = "✅" if success else "❌"
            self.tasks[index]["timestamp"] = datetime.now().strftime("%H:%M:%S")

    def generate_progress_markdown(self) -> str:
        """プログレス表示のMarkdownを生成"""
        lines: List[str] = ["🤖 **AI コード生成プログレス**", ""]

        for task in self.tasks:
            checkbox: str = "[x]" if task["status"] == "completed" else "[ ]"
            line: str = f"- {checkbox} {task['emoji']} {task['description']} _{task['timestamp']}_"
            lines.append(line)

        current_task: Optional[Dict[str, str]] = next(
            (t for t in self.tasks if t["status"] == "pending"), None
        )
        if current_task:
            lines.extend(["", f"**現在のステップ**: {current_task['description']}"])
        else:
            completed_count: int = sum(1 for t in self.tasks if t["status"] == "completed")
            total_count: int = len(self.tasks)
            if completed_count == total_count:
                lines.extend(["", "**ステータス**: 🎉 全て完了"])
            else:
                lines.extend(["", "**ステータス**: ❌ エラーが発生しました"])

        lines.extend(["", f"**開始時刻**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])

        return "\n".join(lines)

    def _make_github_request(
        self, url: str, method: str = "POST", data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """GitHub APIリクエストを実行"""
        if not self.token:
            print("GitHub token not available")
            return None

        headers: Dict[str, str] = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "GitHub-Actions",
        }

        try:
            request_data: Optional[bytes] = None
            if data is not None:
                request_data = json.dumps(data).encode("utf-8")

            req = urllib.request.Request(url, data=request_data, headers=headers, method=method)

            with urllib.request.urlopen(req) as response:
                response_data: bytes = response.read()
                return cast(Dict[str, Any], json.loads(response_data.decode("utf-8")))

        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code}: {e.reason}")
            try:
                error_body: bytes = e.read()
                print(f"Error body: {error_body.decode('utf-8')}")
            except Exception:
                pass
            return None
        except Exception as e:  # pragma: no cover - defensive
            print(f"Request failed: {e}")
            return None

    def post_progress_comment(self) -> None:
        """初回プログレスコメントを投稿"""
        if not self.repo:
            print("Repository information not available")
            return

        url: str = f"https://api.github.com/repos/{self.repo}/issues/{self.issue_number}/comments"
        markdown: str = self.generate_progress_markdown()

        response: Optional[Dict[str, Any]] = self._make_github_request(
            url, "POST", {"body": markdown}
        )

        if response and "id" in response:
            self.comment_id = int(response["id"])
            print(f"Progress comment posted: {self.comment_id}")
        else:
            print("Failed to post progress comment")

    def update_progress_comment(self) -> None:
        """既存のプログレスコメントを更新"""
        if not self.comment_id:
            self.post_progress_comment()
            return

        if not self.repo:
            print("Repository information not available")
            return

        url: str = f"https://api.github.com/repos/{self.repo}/issues/comments/{self.comment_id}"
        markdown: str = self.generate_progress_markdown()

        response: Optional[Dict[str, Any]] = self._make_github_request(
            url, "PATCH", {"body": markdown}
        )

        if response:
            print("Progress comment updated")
        else:
            print("Failed to update progress comment")

    def handle_error(self, index: int, error_message: str) -> None:
        """エラー状態をプログレスに反映"""
        if 0 <= index < len(self.tasks):
            self.tasks[index]["status"] = "failed"
            self.tasks[index]["emoji"] = "❌"
            self.tasks[index]["description"] += f" (エラー: {error_message})"
            self.tasks[index]["timestamp"] = datetime.now().strftime("%H:%M:%S")


def main() -> None:
    """CLI用のメイン関数"""
    if len(sys.argv) < 3:
        print("Usage: python progress_tracker.py <issue_number> <action> [params...]")
        sys.exit(1)

    try:
        issue_number: int = int(sys.argv[1])
    except ValueError:
        print("Error: issue_number must be an integer")
        sys.exit(1)

    action: str = sys.argv[2]

    tracker = ProgressTracker(issue_number)

    if action == "init":
        tracker.add_task("📋 Issue分析開始", "📋")
        tracker.add_task("🔍 コード関連キーワード検証", "🔍")
        tracker.add_task("🤖 AI分析エンジン起動", "🤖")
        tracker.add_task("⚙️ コード生成処理", "⚙️")
        tracker.add_task("📁 ファイル変更検出", "📁")
        tracker.add_task("🌿 ブランチ作成", "🌿")
        tracker.add_task("📝 コミット作成", "📝")
        tracker.add_task("🚀 プルリクエスト生成", "🚀")

        tracker.post_progress_comment()

    elif action == "complete":
        if len(sys.argv) < 4:
            print("Error: complete action requires task_index")
            sys.exit(1)

        try:
            task_index: int = int(sys.argv[3])
        except ValueError:
            print("Error: task_index must be an integer")
            sys.exit(1)

        success: bool = True
        if len(sys.argv) > 4:
            success = sys.argv[4].lower() == "true"

        tracker.complete_task(task_index, success)
        tracker.update_progress_comment()

    elif action == "error":
        if len(sys.argv) < 4:
            print("Error: error action requires task_index")
            sys.exit(1)

        try:
            task_index = int(sys.argv[3])
        except ValueError:
            print("Error: task_index must be an integer")
            sys.exit(1)

        error_msg: str = sys.argv[4] if len(sys.argv) > 4 else "Unknown error"

        tracker.handle_error(task_index, error_msg)
        tracker.update_progress_comment()

    else:
        print(f"Error: Unknown action '{action}'")
        print("Available actions: init, complete, error")
        sys.exit(1)


if __name__ == "__main__":
    main()

