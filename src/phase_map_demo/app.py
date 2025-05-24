"""Phase-map demo – CLI entry point."""

from __future__ import annotations

# ここは元の demo ロジックを呼び出すだけの薄いラッパー
def main() -> None:
    from .phase_map_demo import main_demo
    run_demo()                            #    ↑適宜読み替え

# `python -m phase_map_demo` 用
if __name__ == "__main__":
    main_demo()
