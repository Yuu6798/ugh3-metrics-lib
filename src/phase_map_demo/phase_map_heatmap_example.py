"""Heat-map demo module (auto-generated)"""
import numpy as np
import matplotlib.pyplot as plt

def run_demo() -> None:
    """サンプルの位相マップ・ヒートマップを表示する"""
    x = np.linspace(0, 2*np.pi, 201)
    y = np.linspace(0, 2*np.pi, 201)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(X, Y, Z, shading="auto")
    fig.colorbar(pcm, ax=ax, label="sin(x)·cos(y)")
    ax.set_xlabel("x [rad]")
    ax.set_ylabel("y [rad]")
    ax.set_title("Phase-map heat-map demo")
    plt.show()

__all__ = ["run_demo"]
