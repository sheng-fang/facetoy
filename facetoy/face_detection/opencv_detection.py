"""OpenCV-based face detection implementation."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger

CV_PKG_PATH = Path(cv2.__file__).parent


class FaceDetectorOpenCV:
    """
    OpenCV-based face detector supporting multiple model types.

    This class provides face detection using various OpenCV cascade classifiers
    and DNN models. It returns bounding boxes for detected faces in the input image.
    """

    SUPPORTED_MODELS = {
        "haar_frontalface": CV_PKG_PATH / "data/haarcascade_frontalface_alt.xml",
        "haar_frontalface_default": CV_PKG_PATH / "data/haarcascade_frontalface_default.xml",
        "haar_profileface": CV_PKG_PATH / "data/haarcascade_profileface.xml",
        "lbp_frontalface": CV_PKG_PATH / "data/lbpcascade_frontalface.xml",
    }

    def __init__(self, model_type: str = "haar_frontalface_default", model_path: Optional[Union[str, Path]] = None):
        """
        Initialize the face detector.

        Args:
            model_type (str): Type of model to use. Options:
                - 'haar_frontalface': Haar cascade for frontal faces
                - 'haar_frontalface_default': Default Haar cascade (recommended)
                - 'haar_profileface': Haar cascade for profile faces
                - 'lbp_frontalface': LBP cascade for frontal faces (faster)
            model_path (Optional[str]): Custom path to model file. If None, uses OpenCV's built-in models.

        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If custom model_path is provided but file doesn't exist
        """
        self.model_type = model_type
        self.model_path = model_path if model_path else self.SUPPORTED_MODELS.get(model_type)
        logger.info(f"Using model type: {self.model_type}, model path: {self.model_path}")
        self.detector = self._load_cascade_model()

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. " f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )

    def _load_cascade_model(self) -> cv2.CascadeClassifier:
        """Load Haar or LBP cascade classifier."""
        detector = cv2.CascadeClassifier(str(self.model_path))

        if detector.empty():
            raise RuntimeError(f"Failed to load cascade classifier from: {str(self.model_path)}")

        return detector

    def forward(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the input image and return bounding boxes.

        Args:
            image (np.ndarray): Input image as BGR numpy array
            scale_factor (float): How much the image size is reduced at each scale (for cascade models)
            min_neighbors (int): How many neighbors each face rectangle should retain (for cascade models)
            min_size (Tuple[int, int]): Minimum possible face size. Smaller faces are ignored
            confidence_threshold (float): Confidence threshold for DNN models (0.0 to 1.0)

        Returns:
            List[Tuple[int, int, int, int]]: List of bounding boxes as (x, y, width, height)

        Raises:
            ValueError: If image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")

        if self.model_type.startswith("haar") or self.model_type.startswith("lbp"):
            return self._detect_cascade(image, scale_factor, min_neighbors, min_size)

        return []

    def _detect_cascade(
        self, image: np.ndarray, scale_factor: float, min_neighbors: int, min_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Detect faces using cascade classifier."""
        # Convert to grayscale for cascade detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert to list of tuples
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            dict: Model information including type and path
        """
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "is_loaded": self.detector is not None or self.net is not None,
        }
