from pathlib import Path
from typing import List, Optional, Union

import yaml
from pydantic import BaseModel, Field
from pydantic_yaml import parse_yaml_raw_as

# 1. Define Pydantic Models to match the YAML structure

# --- INFERENCE_CFG Models ---
class FaceDetectorInferenceConfig(BaseModel):
    """
    Pydantic model for the 'face_detector' section within 'INFERENCE_CFG'.
    """

    scale_factor: float
    min_neighbors: int
    min_size: List[int]  # Expects a list of two integers [width, height]


class FunhouseMirrorInferenceConfig(BaseModel):
    """
    Pydantic model for the 'funhouse_mirror_effect' section within 'INFERENCE_CFG'.
    """

    distortion_strength: int
    center_x: Optional[int] = None  # Optional, as it can be null
    center_y: Optional[int] = None  # Optional, as it can be null
    radius: Optional[int] = None  # Optional, as it can be null


class FaceTransformInferenceConfig(BaseModel):
    """
    Pydantic model for the 'face_transform' section within 'INFERENCE_CFG'.
    """

    funhouse_mirror_effect: FunhouseMirrorInferenceConfig


class InferenceConfig(BaseModel):
    """
    Pydantic model for the 'INFERENCE_CFG' section.
    """

    face_detector: FaceDetectorInferenceConfig
    face_transform: FaceTransformInferenceConfig


# --- STATIC_CFG Models ---
class FaceDetectorStaticConfig(BaseModel):
    """
    Pydantic model for the 'face_detector' section within 'STATIC_CFG'.
    """

    wrapper: str
    model_type: str
    model_path: Optional[str] = None  # Optional, as it can be null


class StaticConfig(BaseModel):
    """
    Pydantic model for the 'STATIC_CFG' section.
    """

    face_detector: FaceDetectorStaticConfig


class AppConfig(BaseModel):
    """
    Main Pydantic model for the entire application configuration.
    It uses Field(alias=...) to map YAML keys to model attributes
    if the Python attribute name differs from the YAML key (e.g., casing).
    """

    STATIC_CFG: StaticConfig = Field(alias="STATIC_CFG")
    INFERENCE_CFG: InferenceConfig = Field(alias="INFERENCE_CFG")


def load_config_with_pydantic_yaml(filepath: Union[str, Path] = "default_config.yaml") -> AppConfig:
    """
    Loads a YAML configuration file and validates it against the AppConfig Pydantic model.

    Args:
        filepath (str): The path to the YAML configuration file.

    Returns:
        AppConfig: An instance of the AppConfig class with validated configuration data.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML content.
        pydantic.ValidationError: If the loaded data does not conform to the AppConfig schema.
    """
    try:
        with open(filepath, "r") as file:
            yaml_content = file.read()

        # Use pydantic_yaml to parse and validate the YAML content directly into the model
        config: AppConfig = parse_yaml_raw_as(AppConfig, yaml_content)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{filepath}'")
        raise
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file '{filepath}': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during configuration loading: {e}")
        raise
