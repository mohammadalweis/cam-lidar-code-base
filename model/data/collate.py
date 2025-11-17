"""
Custom collate utilities for sensor samples.
"""

from __future__ import annotations

from typing import List

import torch

from .types import SensorBatch, SensorSample


def sensor_collate_fn(samples: List[SensorSample]) -> SensorBatch:
    """Convert a list of SensorSample objects into batch tensors."""
    if len(samples) == 0:
        raise ValueError("sensor_collate_fn received an empty batch")

    ego = torch.stack([sample.ego_state for sample in samples], dim=0)
    future = torch.stack([sample.future_trajectory for sample in samples], dim=0)
    agents = torch.stack([sample.agents for sample in samples], dim=0)
    agents_mask = torch.stack([sample.agents_mask for sample in samples], dim=0)

    return SensorBatch(
        metadata=[sample.metadata for sample in samples],
        lidar_points=[sample.lidar_points for sample in samples],
        camera_images=[sample.camera_images for sample in samples],
        ego_state=ego,
        future_trajectory=future,
        agents=agents,
        agents_mask=agents_mask,
    )

