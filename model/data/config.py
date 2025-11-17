"""
Configuration helpers for the data block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


@dataclass
class ImageOptions:
    """Image processing configuration."""

    resize_hw: Optional[Sequence[int]] = None  # (H, W)
    antialias: bool = True
    to_float: bool = True


@dataclass
class LidarOptions:
    """Lidar processing configuration."""

    max_points: Optional[int] = 200000
    radius_filter_m: Optional[float] = None


@dataclass
class ScenarioFilterOptions:
    """Wrapper around nuPlan ScenarioFilter arguments."""

    scenario_types: Optional[List[str]] = None
    scenario_tokens: Optional[List[Sequence[str]]] = None
    log_names: Optional[List[str]] = None
    map_names: Optional[List[str]] = None
    num_scenarios_per_type: Optional[int] = None
    limit_total_scenarios: Optional[Union[int, float]] = None
    timestamp_threshold_s: Optional[float] = None
    ego_displacement_minimum_m: Optional[float] = None
    expand_scenarios: bool = True
    remove_invalid_goals: bool = True
    shuffle: bool = True
    ego_start_speed_threshold: Optional[float] = None
    ego_stop_speed_threshold: Optional[float] = None
    speed_noise_tolerance: Optional[float] = None
    token_set_path: Optional[str] = None
    fraction_in_token_set_threshold: Optional[float] = None
    ego_route_radius: Optional[float] = None


@dataclass
class DatasetPaths:
    """Absolute locations used by the dataset block."""

    code_base: Path
    dataset_root: Path
    db_root: Path
    sensor_root: Path
    map_root: Path
    output_dir: Path
    debug_dir: Optional[Path] = None

    def ensure_dirs(self) -> None:
        """Create output and debug directories if needed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.debug_dir:
            self.debug_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DataOptions:
    """Runtime data-processing knobs."""

    map_version: str = "nuplan-maps-v1.0"
    camera_channels: List[str] = field(
        default_factory=lambda: ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
    )
    include_lidar: bool = True
    future_horizon_s: float = 8.0
    future_interval_s: float = 0.5
    max_agents: int = 64
    agent_distance_filter: float = 120.0
    image: ImageOptions = field(default_factory=ImageOptions)
    lidar: LidarOptions = field(default_factory=LidarOptions)

    @property
    def num_future_steps(self) -> int:
        """Return number of trajectory steps to sample."""
        return max(1, int(self.future_horizon_s / self.future_interval_s))


@dataclass
class LoaderOptions:
    """Data loader configuration."""

    batch_size: int = 2
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True


@dataclass
class DatasetConfig:
    """Collected configuration for the dataset block."""

    paths: DatasetPaths
    data: DataOptions
    loader: LoaderOptions
    scenario_filter: ScenarioFilterOptions
    db_files: Optional[List[str]] = None


def _expand_path(path_value: Union[str, Path], base: Optional[Path]) -> Path:
    """Expand user path relative to base, supporting ~."""
    path = Path(path_value).expanduser()
    if not path.is_absolute() and base is not None:
        path = (base / path).resolve()
    return path


def load_dataset_config(path: Union[str, Path]) -> DatasetConfig:
    """Parse YAML file into DatasetConfig."""
    config_path = Path(path).expanduser().resolve()
    raw: Dict[str, Any] = yaml.safe_load(config_path.read_text())

    base_dir = config_path.parent
    paths_raw = raw.get("paths", {})
    code_base = _expand_path(paths_raw.get("code_base", base_dir), base_dir)
    dataset_root = _expand_path(paths_raw.get("dataset_root", base_dir / "dataset"), base_dir)

    db_root = _expand_path(paths_raw.get("db_root", dataset_root), base_dir)
    sensor_root = _expand_path(paths_raw.get("sensor_root", dataset_root), base_dir)
    map_root = _expand_path(paths_raw.get("map_root", dataset_root), base_dir)
    output_dir = _expand_path(paths_raw.get("output_dir", code_base / "model" / "outputs"), base_dir)
    debug_dir_value = paths_raw.get("debug_dir")
    debug_dir = (
        _expand_path(debug_dir_value, base_dir)
        if debug_dir_value is not None
        else output_dir / "data_debug"
    )

    paths = DatasetPaths(
        code_base=code_base,
        dataset_root=dataset_root,
        db_root=db_root,
        sensor_root=sensor_root,
        map_root=map_root,
        output_dir=output_dir,
        debug_dir=debug_dir,
    )

    data_cfg = raw.get("data", {})
    loader_cfg = raw.get("loader", {})
    scenario_filter_cfg = raw.get("scenario_filter", {})

    data_options = DataOptions(
        map_version=data_cfg.get("map_version", "nuplan-maps-v1.0"),
        camera_channels=data_cfg.get(
            "camera_channels", ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"]
        ),
        include_lidar=data_cfg.get("include_lidar", True),
        future_horizon_s=float(data_cfg.get("future_horizon_s", 8.0)),
        future_interval_s=float(data_cfg.get("future_interval_s", 0.5)),
        max_agents=int(data_cfg.get("max_agents", 64)),
        agent_distance_filter=float(data_cfg.get("agent_distance_filter", 120.0)),
        image=ImageOptions(**data_cfg.get("image", {})),
        lidar=LidarOptions(**data_cfg.get("lidar", {})),
    )
    loader_options = LoaderOptions(
        batch_size=int(loader_cfg.get("batch_size", 2)),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        shuffle=bool(loader_cfg.get("shuffle", True)),
    )
    scenario_filter_options = ScenarioFilterOptions(**scenario_filter_cfg)

    config = DatasetConfig(
        paths=paths,
        data=data_options,
        loader=loader_options,
        scenario_filter=scenario_filter_options,
        db_files=raw.get("db_files"),
    )
    paths.ensure_dirs()
    return config
