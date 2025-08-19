"""Keep package init side-effect free to avoid runpy double-execution."""
__all__: list[str] = []
try:
    from importlib.metadata import version
    __version__ = version("ugh3-metrics-lib")
except Exception:
    __version__ = "0"
