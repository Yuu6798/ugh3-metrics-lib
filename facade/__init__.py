"""Keep package init side-effect free to avoid runpy double-execution."""

from __future__ import annotations

# __version__ をできるだけ確実に埋める（失敗時は "0"）
try:
    from importlib.metadata import version as _pkg_version  # type: ignore
    __version__ = _pkg_version("ugh3-metrics-lib")
except Exception:
    try:
        from importlib.metadata import version as _pkg_version  # type: ignore
        __version__ = _pkg_version("por-deltae-lib")
    except Exception:  # pragma: no cover
        __version__ = "0"

from typing import Any, Callable, cast

def collector_main(argv: list[str] | None = None) -> int:
    """Proxy to :func:`facade.collector.main` without importing it eagerly."""
    from .collector import main

    return main(argv)


def run_cycle(*args: Any, **kwargs: Any) -> None:
    """Proxy to :func:`facade.collector.run_cycle` lazily."""
    from .collector import run_cycle as _run_cycle

    fn = cast(Callable[..., None], _run_cycle)
    return fn(*args, **kwargs)


__all__ = ["collector_main", "run_cycle", "__version__"]
