#!/usr/bin/env python3
"""
GitHub Issue Progress Tracker
Issue作成後にプログレスコメントを投稿し、各ステップで動的更新する
"""

import os
import json
import requests
from datetime import datetime
from typing import List, Dict

class ProgressTracker:
    def __init__(self, issue_number: int):
        self.issue_number = issue_number
        self.comment_id = None
        self.tasks = []
        self.repo = os.getenv('GITHUB_REPOSITORY')
        self.token = os.getenv('GITHUB_TOKEN')
        
        if not self.repo or not self.token:
            raise ValueError("GITHUB_REPOSITORY and GITHUB_TOKEN must be set")

    def add_task(self, description: str, emoji: str = "⏳"):
        """タスクをプログレスリストに追加"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.tasks.append({
            'description': description,
            'emoji': emoji,
            'status': 'pending',
            'timestamp': timestamp
        })

    def complete_task(self, index: int, success: bool = True):
        """タスクを完了状態に更新"""
        if 0 <= index < len(self.tasks):
            self.tasks[index]['status'] = 'completed' if success else 'failed'
            self.tasks[index]['emoji'] = '✅' if success else '❌'
            self.tasks[index]['timestamp'] = datetime.now().strftime("%H:%M:%S")

    def generate_progress_markdown(self) -> str:
        """プログレス表示のMarkdownを生成"""
        lines = ["🤖 **AI コード生成プログレス**", ""]

        for i, task in enumerate(self.tasks):
            status_icon = "✅" if task['status'] == 'completed' else "❌" if task['status'] == 'failed' else "⏳"
            checkbox = "[x]" if task['status'] == 'completed' else "[ ]"

            line = f"- {checkbox} {task['emoji']} {task['description']} _{task['timestamp']}_"
            lines.append(line)

        current_task = next((t for t in self.tasks if t['status'] == 'pending'), None)
        if current_task:
            lines.extend(["", f"**現在のステップ**: {current_task['description']}"])
        else:
            completed_count = sum(1 for t in self.tasks if t['status'] == 'completed')
            total_count = len(self.tasks)
            if completed_count == total_count:
                lines.extend(["", "**ステータス**: 🎉 全て完了"])
            else:
                lines.extend(["", "**ステータス**: ❌ エラーが発生しました"])

        lines.extend(["", f"**開始時刻**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])

        return "\n".join(lines)

    def post_progress_comment(self):
        """初回プログレスコメントを投稿"""
        url = f"https://api.github.com/repos/{self.repo}/issues/{self.issue_number}/comments"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

        markdown = self.generate_progress_markdown()

        response = requests.post(url, headers=headers, json={"body": markdown})

        if response.status_code == 201:
            self.comment_id = response.json()["id"]
            print(f"Progress comment posted: {self.comment_id}")
        else:
            print(f"Failed to post comment: {response.status_code}")
            print(response.text)

    def update_progress_comment(self):
        """既存のプログレスコメントを更新"""
        if not self.comment_id:
            self.post_progress_comment()
            return

        url = f"https://api.github.com/repos/{self.repo}/issues/comments/{self.comment_id}"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }

        markdown = self.generate_progress_markdown()

        response = requests.patch(url, headers=headers, json={"body": markdown})

        if response.status_code == 200:
            print("Progress comment updated")
        else:
            print(f"Failed to update comment: {response.status_code}")
            print(response.text)

    def handle_error(self, index: int, error_message: str):
        """エラー状態をプログレスに反映"""
        if 0 <= index < len(self.tasks):
            self.tasks[index]['status'] = 'failed'
            self.tasks[index]['emoji'] = '❌'
            self.tasks[index]['description'] += f" (エラー: {error_message})"
            self.tasks[index]['timestamp'] = datetime.now().strftime("%H:%M:%S")

def main():
    """CLI用のメイン関数"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python progress_tracker.py <issue_number> <action> [params...]")
        sys.exit(1)

    issue_number = int(sys.argv[1])
    action = sys.argv[2]

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
        task_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        success = sys.argv[4].lower() == "true" if len(sys.argv) > 4 else True

        tracker.complete_task(task_index, success)
        tracker.update_progress_comment()

    elif action == "error":
        task_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        error_msg = sys.argv[4] if len(sys.argv) > 4 else "Unknown error"

        tracker.handle_error(task_index, error_msg)
        tracker.update_progress_comment()

if __name__ == "__main__":
    main()
