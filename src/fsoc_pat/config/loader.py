"""Configuration loader — reads YAML/JSON files and merges with defaults.

The loader reads a scenario file, overlays its values onto the defaults,
and returns a fully populated ExperimentConfig. Missing keys use defaults.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import yaml

from fsoc_pat.config.defaults import default_config
from fsoc_pat.config.models import (
    CameraConfig,
    ControllerConfig,
    DetectorConfig,
    DisturbanceConfig,
    EstimatorConfig,
    ExperimentConfig,
    LinkConfig,
    ScenarioConfig,
    TargetConfig,
)
from fsoc_pat.config.validator import validate_config

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML or JSON file.

    Missing keys are filled from defaults. The configuration is
    validated before returning.

    Args:
        path: Path to the configuration file.

    Returns:
        Fully populated and validated ExperimentConfig.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file format is unsupported.
        ConfigurationError: If validation fails.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info("Loading config from: %s", path)

    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    config = _merge_with_defaults(raw)
    validate_config(config)

    logger.info("Config loaded: scenario=%s, seed=%d", config.scenario.name, config.scenario.seed)
    return config


def _merge_with_defaults(raw: dict[str, Any]) -> ExperimentConfig:
    """Merge raw config dict with defaults.

    Only keys present in the raw dict override defaults.

    Args:
        raw: Raw config dictionary from YAML/JSON.

    Returns:
        ExperimentConfig with overrides applied.
    """
    cfg = default_config()

    if "scenario" in raw:
        cfg.scenario = _merge_dataclass(cfg.scenario, raw["scenario"])

    if "camera" in raw:
        cam_raw = raw["camera"]
        # Handle degree → radian conversion for convenience
        if "hfov_deg" in cam_raw:
            cam_raw["hfov_rad"] = math.radians(cam_raw.pop("hfov_deg"))
        if "vfov_deg" in cam_raw:
            cam_raw["vfov_rad"] = math.radians(cam_raw.pop("vfov_deg"))
        cfg.camera = _merge_dataclass(cfg.camera, cam_raw)

    if "target" in raw:
        cfg.target = _merge_dataclass(cfg.target, raw["target"])

    if "detector" in raw:
        cfg.detector = _merge_dataclass(cfg.detector, raw["detector"])

    if "estimator" in raw:
        cfg.estimator = _merge_dataclass(cfg.estimator, raw["estimator"])

    if "controller" in raw:
        cfg.controller = _merge_dataclass(cfg.controller, raw["controller"])

    if "disturbances" in raw:
        cfg.disturbances = _merge_dataclass(cfg.disturbances, raw["disturbances"])

    if "link" in raw:
        link_raw = raw["link"]
        if "beam_divergence_deg" in link_raw:
            link_raw["beam_divergence_rad"] = math.radians(
                link_raw.pop("beam_divergence_deg")
            )
        if "acceptance_angle_deg" in link_raw:
            link_raw["acceptance_angle_rad"] = math.radians(
                link_raw.pop("acceptance_angle_deg")
            )
        cfg.link = _merge_dataclass(cfg.link, link_raw)

    return cfg


def _merge_dataclass(obj: Any, overrides: dict[str, Any]) -> Any:
    """Merge a dict of overrides into a dataclass instance.

    Creates a new instance with overridden fields.
    Unknown keys are warned and ignored.

    Args:
        obj: Existing dataclass instance.
        overrides: Dictionary of field overrides.

    Returns:
        New dataclass instance with overrides applied.
    """
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(obj)}
    kwargs = {}

    for key, value in overrides.items():
        if key in field_names:
            # Convert lists to tuples for tuple-typed fields
            current = getattr(obj, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value
        else:
            logger.warning("Unknown config key '%s' in %s — ignored", key, type(obj).__name__)

    return dataclasses.replace(obj, **kwargs)
