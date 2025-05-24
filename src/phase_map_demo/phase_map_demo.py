from __future__ import annotations



def run_demo() -> None:

    """CLI entry: just run the example."""

    from .phase_map_heatmap_example import run_demo

    phase_map_heatmap_example()

from .cli import main

def main_demo() -> None:
    """CLI entry: just run the example."""
    from .phase_map_heatmap_example import run_demo
    run_demo()

