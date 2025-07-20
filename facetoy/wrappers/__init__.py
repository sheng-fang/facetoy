"""Face detection module."""
from typing import Union

from .opencv_detection import FaceDetectorOpenCV

WRAPPERS = {
    "FaceDetectorOpenCV": FaceDetectorOpenCV,
}

FaceDetectorType = Union[FaceDetectorOpenCV]


__all__ = ["WRAPPERS", "FaceDetectorOpenCV"]
