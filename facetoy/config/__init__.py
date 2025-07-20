from pathlib import Path

from facetoy.config.base import load_config_with_pydantic_yaml

default_config_path = Path(__file__).parent / "default_config.yaml"

config = load_config_with_pydantic_yaml(default_config_path)

default_static_cfg = config.STATIC_CFG
default_inference_cfg = config.INFERENCE_CFG
