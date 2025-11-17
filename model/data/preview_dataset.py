"""
CLI utility to inspect one dataset sample and dump debug visualizations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PilImage

from .config import DatasetConfig, load_dataset_config
from .nuplan_dataset import NuPlanSensorDataset


def _save_camera_images(output_dir: Path, sample, prefix: str) -> None:
    """Persist camera tensors as png files."""
    for name, tensor in sample.camera_images.items():
        array = tensor.detach().cpu().numpy()
        array = np.transpose(array, (1, 2, 0))
        if array.max() <= 1.0:
            array = (array * 255.0).clip(0, 255)
        pil = PilImage.fromarray(array.astype(np.uint8))
        path = output_dir / f"{prefix}_{name}.png"
        pil.save(path)


def _save_lidar_plot(output_dir: Path, sample, prefix: str) -> None:
    """Create a top-down scatter plot of the lidar points and future trajectory."""
    lidar = sample.lidar_points.cpu().numpy()
    plt.figure(figsize=(6, 6))
    if lidar.size > 0:
        plt.scatter(lidar[:, 0], lidar[:, 1], s=0.1, alpha=0.5, c=lidar[:, 3], cmap="viridis")
    future = sample.future_trajectory.cpu().numpy()
    plt.plot(future[:, 0], future[:, 1], c="red", linewidth=2, label="ego_future")
    plt.axis("equal")
    plt.title(f"Scenario {sample.metadata.scenario_token}")
    plt.legend()
    (output_dir / f"{prefix}_lidar.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{prefix}_lidar.png", bbox_inches="tight")
    plt.close()


def _dump_metadata(output_dir: Path, sample, prefix: str) -> None:
    """Write metadata summary."""
    payload = {
        "scenario_token": sample.metadata.scenario_token,
        "log_name": sample.metadata.log_name,
        "scenario_type": sample.metadata.scenario_type,
        "timestamp_us": sample.metadata.timestamp_us,
        "map_name": sample.metadata.map_name,
        "num_lidar_points": int(sample.lidar_points.shape[0]),
        "cameras": list(sample.camera_images.keys()),
        "num_agents": int(sample.agents_mask.sum().item()),
    }
    with (output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def preview_sample(config: DatasetConfig, index: int) -> None:
    """Load dataset and export debug artifacts for a single sample."""
    dataset = NuPlanSensorDataset(config)
    sample = dataset[index]
    output_dir = config.paths.debug_dir or (config.paths.output_dir / "data_debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{index:04d}_{sample.metadata.scenario_token}"

    _save_camera_images(output_dir, sample, prefix)
    _save_lidar_plot(output_dir, sample, prefix)
    _dump_metadata(output_dir, sample, prefix)
    print(f"Saved debug artifacts to {output_dir} for sample #{index}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview nuPlan dataset sample.")
    parser.add_argument("--config", type=str, default="model/data/dataset_config.yaml", help="Path to dataset YAML.")
    parser.add_argument("--index", type=int, default=0, help="Sample index to preview.")
    args = parser.parse_args()

    cfg = load_dataset_config(args.config)
    preview_sample(cfg, args.index)


if __name__ == "__main__":
    main()
