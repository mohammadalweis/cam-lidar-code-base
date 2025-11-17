"""
Type helpers for the data module.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class SampleMetadata:
    """Metadata describing a single dataset element."""

    scenario_token: str
    log_name: str
    scenario_type: str
    timestamp_us: int
    map_name: Optional[str] = None


@dataclass
class SensorSample:
    """Container returned by the dataset __getitem__."""

    metadata: SampleMetadata
    lidar_points: torch.Tensor
    camera_images: Dict[str, torch.Tensor]
    ego_state: torch.Tensor
    future_trajectory: torch.Tensor
    agents: torch.Tensor
    agents_mask: torch.Tensor


@dataclass
class SensorBatch:
    """Batched version used by the collate_fn."""

    metadata: List[SampleMetadata]
    lidar_points: List[torch.Tensor]
    camera_images: List[Dict[str, torch.Tensor]]
    ego_state: torch.Tensor
    future_trajectory: torch.Tensor
    agents: torch.Tensor
    agents_mask: torch.Tensor

