from pathlib import Path

from facetoy.config.base import load_config_with_pydantic_yaml

config_path = Path(__file__).parent.parent / "facetoy/config/default_config.yaml"


def test_load_config_with_pydantic_yaml() -> None:
    config = load_config_with_pydantic_yaml(config_path)
    assert config is not None
    assert config.STATIC_CFG is not None
    assert config.INFERENCE_CFG is not None
