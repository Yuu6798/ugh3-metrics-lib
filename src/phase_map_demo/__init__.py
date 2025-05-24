"""phase_map_demo – サンプル CLI 付きパッケージ"""

def main() -> None:
    # import は関数内で行い循環を防ぐ
    from .cli import main as _main
    _main()

__all__ = ["main"]
