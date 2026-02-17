"""OmegaConf-based configuration loader for AutoRisk-RM."""

from pathlib import Path
from typing import Any
from importlib import resources as importlib_resources

from omegaconf import DictConfig, OmegaConf

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
_DEFAULT_CONFIG_RESOURCE_PACKAGE = "autorisk.resources.configs"
_DEFAULT_CONFIG_RESOURCE_NAME = "default.yaml"


def _load_default_config() -> DictConfig:
    if _DEFAULT_CONFIG.exists():
        return OmegaConf.load(str(_DEFAULT_CONFIG))
    resource = importlib_resources.files(_DEFAULT_CONFIG_RESOURCE_PACKAGE).joinpath(_DEFAULT_CONFIG_RESOURCE_NAME)
    with importlib_resources.as_file(resource) as path:
        return OmegaConf.load(str(path))


def load_config(
    overrides: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> DictConfig:
    """Load configuration from YAML with optional overrides.

    Priority (highest first):
        1. CLI / programmatic overrides
        2. Custom config_path YAML
        3. configs/default.yaml

    Args:
        overrides: Dict of dot-notation overrides (e.g. {"mining.top_n": 50}).
        config_path: Path to a custom YAML config to merge on top of defaults.

    Returns:
        Merged OmegaConf DictConfig.
    """
    base = _load_default_config()

    if config_path is not None:
        custom = OmegaConf.load(str(config_path))
        base = OmegaConf.merge(base, custom)

    if overrides:
        override_conf = OmegaConf.create(overrides)
        base = OmegaConf.merge(base, override_conf)

    OmegaConf.resolve(base)
    return base
