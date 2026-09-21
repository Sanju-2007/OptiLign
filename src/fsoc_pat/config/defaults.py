"""Default configuration values.

Every default is a documented design assumption. Where the project
requirements do not define a numerical value, the default is clearly
labeled as such.

These defaults can be overridden by scenario YAML files or CLI arguments.
"""

from __future__ import annotations

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


def default_config() -> ExperimentConfig:
    """Return the default experiment configuration.

    All values are design assumptions unless noted otherwise.

    Returns:
        ExperimentConfig with all defaults applied.
    """
    return ExperimentConfig(
        scenario=ScenarioConfig(
            name="default",
            duration_s=10.0,       # Design assumption: 10 s nominal scenario
            timestep_s=0.01,       # Design assumption: 100 Hz loop rate
            seed=42,
        ),
        camera=CameraConfig(),
        target=TargetConfig(),
        detector=DetectorConfig(),
        estimator=EstimatorConfig(),
        controller=ControllerConfig(),
        disturbances=DisturbanceConfig(),
        link=LinkConfig(),
    )
