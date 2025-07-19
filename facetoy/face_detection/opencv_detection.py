"""OpenCV-based face detection implementation."""

import cv2
import numpy as np






from typing import List, Tuple, Optional
from pathlib import Path


class FaceDetectorOpenCV:
    """
    OpenCV-based face detector supporting multiple model types.

    This class provides face detection using various OpenCV cascade classifiers
    and DNN models. It returns bounding boxes for detected faces in the input image.
    """

    SUPPORTED_MODELS = {
        "haar_frontalface": "haarcascade_frontalface_alt.xml",
        "haar_frontalface_default": "haarcascade_frontalface_default.xml",
        "haar_profileface": "haarcascade_profileface.xml",
        "lbp_frontalface": "lbpcascade_frontalface.xml",
        "dnn_caffe": "opencv_face_detector.caffemodel",
        "dnn_tensorflow": "opencv_face_detector_uint8.pb",
    }

    def __init__(self, model_type: str = "haar_frontalface_default", model_path: Optional[str] = None):
        """
        Initialize the face detector.

        Args:
            model_type (str): Type of model to use. Options:
                - 'haar_frontalface': Haar cascade for frontal faces
                - 'haar_frontalface_default': Default Haar cascade (recommended)
                - 'haar_profileface': Haar cascade for profile faces
                - 'lbp_frontalface': LBP cascade for frontal faces (faster)
                - 'dnn_caffe': DNN model using Caffe framework
                - 'dnn_tensorflow': DNN model using TensorFlow
            model_path (Optional[str]): Custom path to model file. If None, uses OpenCV's built-in models.

        Raises:
            ValueError: If model_type is not supported
            FileNotFoundError: If custom model_path is provided but file doesn't exist
        """
        self.model_type = model_type
        self.model_path = model_path
        self.detector = self._load_cascade_model()
        self.net = self._load_dnn_model()

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. " f"Supported types: {list(self.SUPPORTED_MODELS.keys())}"
            )

    def _load_cascade_model(self) -> cv2.CascadeClassifier:
        """Load Haar or LBP cascade classifier."""
        if self.model_path:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            cascade_path = self.model_path
        else:
            # Use OpenCV's built-in cascade files
            cascade_file = self.SUPPORTED_MODELS[self.model_type]
            cascade_path = cv2.data.haarcascades + cascade_file

            # Fallback to default if specific cascade not found
            if not Path(cascade_path).exists():
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        detector = cv2.CascadeClassifier(cascade_path)

        if detector.empty():
            raise RuntimeError(f"Failed to load cascade classifier from: {cascade_path}")

        return detector

    def _load_dnn_model(self) -> cv2.dnn.Net:
        """Load DNN-based face detection model."""
        if not self.model_path:
            raise ValueError("DNN models require explicit model_path parameter")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"DNN model file not found: {self.model_path}")

        if self.model_type == "dnn_caffe":
            # For Caffe models, also need prototxt file
            prototxt_path = str(Path(self.model_path).with_suffix(".prototxt"))
            if not Path(prototxt_path).exists():
                raise FileNotFoundError(f"Prototxt file not found: {prototxt_path}")
            net = cv2.dnn.readNetFromCaffe(prototxt_path, self.model_path)
        elif self.model_type == "dnn_tensorflow":
            net = cv2.dnn.readNetFromTensorflow(self.model_path)

        return net

    def forward(
        self,
        image: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        confidence_threshold: float = 0.5,
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
        elif self.model_type.startswith("dnn"):
            return self._detect_dnn(image, confidence_threshold)

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

    def _detect_dnn(self, image: np.ndarray, confidence_threshold: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN model."""
        h, w = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])

        # Set input to the network
        self.net.setInput(blob)

        # Run forward pass
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > confidence_threshold:
                # Get bounding box coordinates
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)

                # Convert to (x, y, width, height) format
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces

    def detect_and_draw(
        self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2, **kwargs
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Detect faces and draw bounding boxes on the image.

        Args:
            image (np.ndarray): Input image
            color (Tuple[int, int, int]): BGR color for bounding box
            thickness (int): Thickness of bounding box lines
            **kwargs: Additional arguments passed to forward()

        Returns:
            Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
                (Image with drawn boxes, List of bounding boxes)
        """
        faces = self.forward(image, **kwargs)
        result_image = image.copy()

        for x, y, w, h in faces:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)

        return result_image, faces

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
