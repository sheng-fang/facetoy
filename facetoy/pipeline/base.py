from abc import ABC, abstractmethod
from typing import Any, Self

from facetoy.config.base import InferenceConfig, StaticConfig


class PipeBase(ABC):
    """Base class for pipeline components."""

    @classmethod
    @abstractmethod
    def load(cls, config: StaticConfig) -> Self:
        """Load a pipeline component from the configuration."""
        pass

    @abstractmethod
    def forward(self, inputs: Any, inference_cfg: InferenceConfig) -> Any:
        """Process the input data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, inputs: Any, inference_cfg: InferenceConfig) -> Any:
        """Invoke the pipeline component."""
        return self.forward(inputs, inference_cfg)
