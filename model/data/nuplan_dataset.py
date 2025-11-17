"""
PyTorch dataset that exposes nuPlan sensor data for camera+LiDAR experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image as PilImage
from torch.utils.data import Dataset

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.database.utils.image import Image
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.observation.observation_type import (
    CameraChannel,
    LidarChannel,
    SensorChannel,
)
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

from .config import DatasetConfig, ScenarioFilterOptions
from .types import SampleMetadata, SensorSample

logger = logging.getLogger(__name__)


def _scenario_filter_from_options(options: ScenarioFilterOptions) -> ScenarioFilter:
    """Convert helper options into nuPlan ScenarioFilter."""
    token_set_path = Path(options.token_set_path).expanduser() if options.token_set_path else None
    return ScenarioFilter(
        scenario_types=options.scenario_types,
        scenario_tokens=options.scenario_tokens,
        log_names=options.log_names,
        map_names=options.map_names,
        num_scenarios_per_type=options.num_scenarios_per_type,
        limit_total_scenarios=options.limit_total_scenarios,
        timestamp_threshold_s=options.timestamp_threshold_s,
        ego_displacement_minimum_m=options.ego_displacement_minimum_m,
        expand_scenarios=options.expand_scenarios,
        remove_invalid_goals=options.remove_invalid_goals,
        shuffle=options.shuffle,
        ego_start_speed_threshold=options.ego_start_speed_threshold,
        ego_stop_speed_threshold=options.ego_stop_speed_threshold,
        speed_noise_tolerance=options.speed_noise_tolerance,
        token_set_path=token_set_path,
        fraction_in_token_set_threshold=options.fraction_in_token_set_threshold,
        ego_route_radius=options.ego_route_radius,
    )


def _ego_state_to_tensor(state: EgoState) -> torch.Tensor:
    """Convert EgoState to a flat tensor."""
    dyn = state.dynamic_car_state
    return torch.tensor(
        [
            state.rear_axle.x,
            state.rear_axle.y,
            state.rear_axle.heading,
            dyn.rear_axle_velocity_2d.x,
            dyn.rear_axle_velocity_2d.y,
            dyn.rear_axle_acceleration_2d.x,
            dyn.rear_axle_acceleration_2d.y,
            state.tire_steering_angle,
            state.time_point.time_us * 1e-6,
        ],
        dtype=torch.float32,
    )


def _camera_image_to_tensor(
    image: Image,
    resize_hw: Optional[Sequence[int]],
    antialias: bool,
    normalize: bool,
) -> torch.Tensor:
    """Convert camera image object into CHW tensor."""
    if image is None:
        raise ValueError("Camera image is None")

    pil_img: PilImage.Image = image.as_pil
    if resize_hw:
        height, width = int(resize_hw[0]), int(resize_hw[1])
        pil_img = pil_img.resize((width, height), PilImage.BILINEAR if antialias else PilImage.NEAREST)
    array = np.asarray(pil_img, dtype=np.float32)
    if normalize:
        array /= 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def _lidar_pointcloud_to_tensor(
    pointcloud: Optional[LidarPointCloud],
    max_points: Optional[int],
) -> torch.Tensor:
    """Convert lidar pointcloud to tensor of shape (N, F)."""
    if pointcloud is None:
        return torch.zeros((0, 6), dtype=torch.float32)

    points = np.asarray(pointcloud.points.T, dtype=np.float32)
    if max_points is not None and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return torch.from_numpy(points)


def _agents_to_tensor(
    tracked_objects: TrackedObjects,
    max_agents: int,
    ego_xy: Tuple[float, float],
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert tracked objects to dense tensor with padding.
    Returns (agents_tensor, mask_tensor).
    """
    agent_rows: List[Tuple[float, ...]] = []
    for obj in tracked_objects.get_agents():
        if obj.tracked_object_type == TrackedObjectType.EGO:
            continue
        box = obj.box
        rel_x = box.center.x - ego_xy[0]
        rel_y = box.center.y - ego_xy[1]
        distance = float(np.hypot(rel_x, rel_y))
        if distance > distance_threshold:
            continue
        dims = box.dimensions
        agent_rows.append(
            (
                rel_x,
                rel_y,
                box.center.heading,
                dims.length,
                dims.width,
                dims.height,
                obj.velocity.x,
                obj.velocity.y,
                obj.angular_velocity or 0.0,
                float(int(obj.tracked_object_type)),
                distance,
            )
        )

    agent_rows.sort(key=lambda row: row[-1])
    feature_dim = 10
    padded = np.zeros((max_agents, feature_dim), dtype=np.float32)
    mask = np.zeros((max_agents,), dtype=np.float32)
    for idx, row in enumerate(agent_rows[:max_agents]):
        padded[idx, :] = row[:-1]
        mask[idx] = 1.0

    return torch.from_numpy(padded), torch.from_numpy(mask)


@dataclass
class DatasetStats:
    """Simple tracker for dataset creation."""

    num_scenarios: int
    camera_channels: List[str]


class NuPlanSensorDataset(Dataset[SensorSample]):
    """Dataset yielding synchronized camera + LiDAR nuPlan samples."""

    def __init__(self, config: DatasetConfig, verbose: bool = True):
        self.cfg = config
        self._camera_channels = [CameraChannel[name] for name in config.data.camera_channels]
        self._sensor_channels: List[SensorChannel] = list(self._camera_channels)
        if config.data.include_lidar:
            self._sensor_channels.append(LidarChannel.MERGED_PC)

        scenario_builder = NuPlanScenarioBuilder(
            data_root=str(config.paths.db_root),
            map_root=str(config.paths.map_root),
            sensor_root=str(config.paths.sensor_root),
            db_files=config.db_files,
            map_version=config.data.map_version,
            include_cameras=len(self._camera_channels) > 0,
            verbose=verbose,
        )
        filter_options = _scenario_filter_from_options(config.scenario_filter)
        worker = Sequential()
        self._scenarios: List[NuPlanScenario] = scenario_builder.get_scenarios(filter_options, worker)
        if len(self._scenarios) == 0:
            raise RuntimeError("No scenarios matched the provided filter. Check your dataset paths and filter config.")

        self.stats = DatasetStats(
            num_scenarios=len(self._scenarios),
            camera_channels=[channel.name for channel in self._camera_channels],
        )
        logger.info(
            "Loaded %s scenarios (cameras=%s) from %s",
            self.stats.num_scenarios,
            self.stats.camera_channels,
            config.paths.db_root,
        )

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(self, idx: int) -> SensorSample:
        scenario = self._scenarios[idx]
        sensors = scenario.get_sensors_at_iteration(0, channels=self._sensor_channels)
        lidar_tensor = _lidar_pointcloud_to_tensor(
            sensors.pointcloud[LidarChannel.MERGED_PC] if sensors.pointcloud else None,
            max_points=self.cfg.data.lidar.max_points,
        )

        camera_tensors: Dict[str, torch.Tensor] = {}
        if sensors.images:
            for channel, image in sensors.images.items():
                if image is None:
                    continue
                camera_tensors[channel.name] = _camera_image_to_tensor(
                    image=image,
                    resize_hw=self.cfg.data.image.resize_hw,
                    antialias=self.cfg.data.image.antialias,
                    normalize=self.cfg.data.image.to_float,
                )

        ego_state = scenario.get_ego_state_at_iteration(0)
        ego_tensor = _ego_state_to_tensor(ego_state)

        future_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0,
                time_horizon=self.cfg.data.future_horizon_s,
                num_samples=self.cfg.data.num_future_steps,
            )
        )
        future_tensor = torch.zeros((self.cfg.data.num_future_steps, ego_tensor.shape[0]), dtype=torch.float32)
        for i in range(min(len(future_states), self.cfg.data.num_future_steps)):
            future_tensor[i] = _ego_state_to_tensor(future_states[i])

        tracked_objects = scenario.get_tracked_objects_at_iteration(0).tracked_objects
        agents_tensor, agents_mask = _agents_to_tensor(
            tracked_objects=tracked_objects,
            max_agents=self.cfg.data.max_agents,
            ego_xy=(ego_state.rear_axle.x, ego_state.rear_axle.y),
            distance_threshold=self.cfg.data.agent_distance_filter,
        )

        metadata = SampleMetadata(
            scenario_token=scenario.token,
            log_name=scenario.log_name,
            scenario_type=scenario.scenario_type,
            timestamp_us=scenario.get_time_point(0).time_us,
            map_name=getattr(scenario, "_map_name", None),
        )

        return SensorSample(
            metadata=metadata,
            lidar_points=lidar_tensor,
            camera_images=camera_tensors,
            ego_state=ego_tensor,
            future_trajectory=future_tensor,
            agents=agents_tensor,
            agents_mask=agents_mask,
        )
