"""Offline three-panel animation; no live dashboard dependencies."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def write_demo_gif(
    frames: list[tuple[np.ndarray, list[tuple[float, float]], list[str]]], path: Path
) -> None:
    """Compose overhead pixels, titration curve, and safety/tool log.

    Args:
        frames: Camera frames paired with observations and safety messages.
        path: Output GIF path.
    """
    images: list[Any] = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=80)
    for shot, points, logs in frames:
        for ax in axes:
            ax.clear()
        axes[0].imshow(shot)
        axes[0].set_title("SIM: gantry pipetting")
        axes[0].axis("off")
        x, y = zip(*points, strict=True)
        axes[1].plot(x, y, "o-", color="#bf2863")
        axes[1].set(xlabel="NaOH added (uL)", ylabel="pH", ylim=(2, 13), xlim=(0, 50))
        axes[1].set_title("Measured titration curve")
        axes[1].grid(alpha=0.2)
        import textwrap

        text = "\n\n".join(textwrap.fill(line, 43) for line in logs[:4])
        text += "\n\n" + "\n".join(logs[-2:])
        axes[2].text(0, 1, text, va="top", fontsize=7, transform=axes[2].transAxes)
        axes[2].axis("off")
        axes[2].set_title("Safety gate / tool log")
        fig.tight_layout()
        fig.canvas.draw()
        images.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(
                colors=128
            )
        )
    plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:], duration=110, loop=0)
