from dataclasses import dataclass
from functools import partial
from typing import List, Self

import numpy as np
from loguru import logger

from facetoy.config.base import InferenceConfig, StaticConfig
from facetoy.pipeline.base import PipeBase
from facetoy.utils import FACE_TRANSFORMS
from facetoy.wrappers import WRAPPERS, FaceDetectorType
from facetoy.wrappers.base import FacePrediction


@dataclass
class PipeResultFaceTransform:
    image_original: np.ndarray
    image_transformed: np.ndarray
    face_predictions: List[FacePrediction]


class PipeFaceTransform(PipeBase):
    """Pipeline component for face transformation effects."""

    def __init__(self, face_detector: FaceDetectorType) -> None:
        self.face_detector: FaceDetectorType = face_detector

    @classmethod
    def load(cls, config: StaticConfig) -> Self:
        if face_detector_type := WRAPPERS.get(config.face_detector.wrapper, None):
            kwargs = config.face_detector.model_dump()
            kwargs.pop("wrapper", None)  # Remove wrapper key
            face_detector = face_detector_type(**kwargs)
        else:
            raise ValueError(f"Unsupported face detector wrapper: {config.face_detector.wrapper}")

        return cls(
            face_detector=face_detector,
        )

    def forward(self, inputs: List[np.ndarray], inference_cfg: InferenceConfig) -> List[PipeResultFaceTransform]:
        infer_dict = inference_cfg.face_transform.model_dump()

        face_transform = None
        for k, v in infer_dict.items():
            if v.get("use"):
                infer_params = v.copy()
                infer_params.pop("use", None)
                print(f"Using face transform: {k} with params: {infer_params}")
                if face_transform := FACE_TRANSFORMS.get(k):
                    face_transform = partial(face_transform, **infer_params)
                break

        logger.debug(f"Using face transform: {face_transform if face_transform else 'None'}")

        outputs = []

        for image in inputs:
            # Detect faces in the image
            face_predictions = self.face_detector(image, **inference_cfg.face_detector.model_dump())
            img_copy = image.copy()
            # Apply funhouse mirror effect to each detected face
            for prediction in face_predictions:
                if face_transform:
                    x, y, w, h = prediction.bbox_xywh
                    face = image[y : y + h, x : x + w]
                    face_transformed = face_transform(face)
                    img_copy[y : y + h, x : x + w] = face_transformed
            outputs.append(
                PipeResultFaceTransform(
                    image_original=image,
                    image_transformed=img_copy,
                    face_predictions=face_predictions,
                )
            )

        return outputs
