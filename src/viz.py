"""Visualization utilities: animated GIF and ablation table PNG."""

from typing import Dict, List

import numpy as np
from PIL import Image


def make_demo_gif(frames: List[np.ndarray], path: str, fps: int = 10) -> None:
    """Save a list of RGB numpy frames as an animated GIF.

    frames: list of (H, W, 3) uint8 numpy arrays — each is one video frame
    path:   output path, e.g. 'assets/demo.gif'
    fps:    frames per second (default 10)
    """
    pil_frames  = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    duration_ms = int(1000 / fps)
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"GIF saved → {path} ({len(frames)} frames at {fps} fps)")


def plot_ablation_table(results: Dict[str, Dict[str, float]], path: str) -> None:
    """Save ablation results as a matplotlib table PNG.

    results: dict of {variant_name: {metric_name: float_value}}
    Example:
        {
            "Full model":   {"pred_loss": 0.12, "idm_acc": 0.89},
            "No IDM":       {"pred_loss": 0.19, "idm_acc": 0.50},
            "No VICReg":    {"pred_loss": 0.31, "idm_acc": 0.55},
        }
    path: output path for the PNG file
    """
    import matplotlib.pyplot as plt

    variants    = list(results.keys())
    metrics     = list(results[variants[0]].keys())
    table_data  = [[f"{results[v][m]:.4f}" for m in metrics] for v in variants]

    fig, ax = plt.subplots(figsize=(max(6, len(metrics) * 2), len(variants) + 1))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        rowLabels=variants,
        colLabels=metrics,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    plt.title("Ablation Study", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Ablation table saved → {path}")
