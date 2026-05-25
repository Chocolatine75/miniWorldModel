"""Visualization utilities: animated GIF and ablation table PNG."""

from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw


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


def make_comparison_gif(
    frames_random: List[np.ndarray],
    frames_cem: List[np.ndarray],
    path: str,
    fps: int = 10,
) -> None:
    """Save a side-by-side comparison GIF of a random-policy episode vs a CEM-planner episode.

    frames_random: list of (H, W, 3) uint8 numpy arrays for the random policy
    frames_cem:    list of (H, W, 3) uint8 numpy arrays for the CEM planner
    path:          output path, e.g. 'assets/comparison.gif'
    fps:           frames per second (default 10)

    Both sequences play simultaneously; if one is shorter its last frame is repeated.
    Text labels "Random Policy" and "CEM Planner" are drawn in white on a dark banner
    at the top of each panel.
    """
    LABEL_HEIGHT = 20   # pixels reserved for the text banner
    TEXT_COLOR   = (255, 255, 255)
    BG_COLOR     = (30, 30, 30)

    n_frames = max(len(frames_random), len(frames_cem))

    def _to_pil(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8))

    # Determine a common panel height (max of the two sequences' first-frame heights).
    h_rand = frames_random[0].shape[0]
    h_cem  = frames_cem[0].shape[0]
    panel_h = max(h_rand, h_cem)

    def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        scale = target_h / img.height
        new_w = max(1, int(img.width * scale))
        return img.resize((new_w, target_h), Image.BILINEAR)

    combined_frames: List[Image.Image] = []

    for i in range(n_frames):
        # Clamp indices so the last frame is repeated when one sequence is shorter.
        idx_r = min(i, len(frames_random) - 1)
        idx_c = min(i, len(frames_cem) - 1)

        pil_r = _resize_to_height(_to_pil(frames_random[idx_r]), panel_h)
        pil_c = _resize_to_height(_to_pil(frames_cem[idx_c]),    panel_h)

        total_w = pil_r.width + pil_c.width
        total_h = panel_h + LABEL_HEIGHT

        # Build the combined frame.
        combined = Image.new("RGB", (total_w, total_h), BG_COLOR)

        # Paste panels below the label banner.
        combined.paste(pil_r, (0,           LABEL_HEIGHT))
        combined.paste(pil_c, (pil_r.width, LABEL_HEIGHT))

        # Draw dark banner and labels.
        draw = ImageDraw.Draw(combined)
        # Left banner background (already BG_COLOR from Image.new fill).
        # Right banner background.
        draw.rectangle(
            [pil_r.width, 0, total_w - 1, LABEL_HEIGHT - 1],
            fill=BG_COLOR,
        )
        # Label text — centered in each panel.
        draw.text(
            (pil_r.width // 2, LABEL_HEIGHT // 2),
            "Random Policy",
            fill=TEXT_COLOR,
            anchor="mm",
        )
        draw.text(
            (pil_r.width + pil_c.width // 2, LABEL_HEIGHT // 2),
            "CEM Planner",
            fill=TEXT_COLOR,
            anchor="mm",
        )

        combined_frames.append(combined)

    duration_ms = int(1000 / fps)
    combined_frames[0].save(
        path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Comparison GIF saved → {path} ({n_frames} frames at {fps} fps)")


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
