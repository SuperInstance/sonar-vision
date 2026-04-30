"""
Underwater sonar visualization tools.

Simple matplotlib-based plotting for sonar sweeps, detection overlays,
and training output comparisons.
"""

from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_sonar_sweep(
    sweep_array: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot a sonar sweep as a matplotlib heatmap.

    Args:
        sweep_array: 2-D array of shape (bearing_bins, depth_bins) with
            sonar intensity values.
        save_path: If provided, the figure is saved to this path instead of
            being shown interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        sweep_array.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )

    ax.set_xlabel("Bearing (bins)")
    ax.set_ylabel("Depth (bins)")
    ax.set_title("Sonar Sweep Intensity")
    fig.colorbar(im, ax=ax, label="Intensity")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)


def plot_detection_overlay(
    sweep_array: np.ndarray,
    detections: List[Dict],
    save_path: Optional[str] = None,
) -> None:
    """Plot a sonar sweep with detection circles overlaid.

    Args:
        sweep_array: 2-D array of shape (bearing_bins, depth_bins).
        detections: List of detection dicts. Each dict should contain at
            least ``bearing``, ``depth``, and optionally ``radius`` (in bin
            units; defaults to 5).
        save_path: If provided, the figure is saved to this path instead of
            being shown interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        sweep_array.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Intensity")

    for det in detections:
        b = det.get("bearing", 0)
        d = det.get("depth", 0)
        r = det.get("radius", 5)

        circle = plt.Circle(
            (b, d),
            r,
            color="red",
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(circle)

    ax.set_xlabel("Bearing (bins)")
    ax.set_ylabel("Depth (bins)")
    ax.set_title("Sonar Sweep with Detections")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)


def plot_training_comparison(
    predicted_frame: np.ndarray,
    camera_frame: np.ndarray,
    depth_map: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plot a side-by-side training comparison.

    Displays three panels: predicted camera frame, ground-truth camera frame,
    and predicted depth map.

    Args:
        predicted_frame: Array of shape (H, W) or (H, W, C).
        camera_frame: Array of shape (H, W) or (H, W, C).
        depth_map: Array of shape (H, W).
        save_path: If provided, the figure is saved to this path instead of
            being shown interactively.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Predicted frame
    ax = axes[0]
    if predicted_frame.ndim == 2:
        ax.imshow(predicted_frame, cmap="gray")
    else:
        ax.imshow(predicted_frame)
    ax.set_title("Predicted Frame")
    ax.axis("off")

    # Camera frame
    ax = axes[1]
    if camera_frame.ndim == 2:
        ax.imshow(camera_frame, cmap="gray")
    else:
        ax.imshow(camera_frame)
    ax.set_title("Camera Frame")
    ax.axis("off")

    # Depth map
    ax = axes[2]
    im = ax.imshow(depth_map, cmap="plasma")
    ax.set_title("Depth Map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
