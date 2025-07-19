"""Camera operations for capturing video streams."""

from typing import Generator, List, Optional

import cv2
import numpy as np


def get_camera_stream(
    camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None, fps: Optional[int] = None
) -> Generator[np.ndarray, None, None]:
    """
    Generate images from camera stream one by one.

    This function creates a generator that yields camera frames continuously.
    It captures video from the specified camera device and yields each frame
    as a numpy array.

    Args:
        camera_index (int): Index of the camera device (default: 0 for default camera)
        width (Optional[int]): Desired frame width in pixels. If None, uses camera default.
        height (Optional[int]): Desired frame height in pixels. If None, uses camera default.
        fps (Optional[int]): Desired frames per second. If None, uses camera default.

    Yields:
        np.ndarray: Camera frame as BGR image array with shape (height, width, 3)

    Raises:
        RuntimeError: If camera cannot be opened or initialized

    Example:
        >>> from facetoy.io import get_camera_stream
        >>>
        >>> # Basic usage - get frames from default camera
        >>> for frame in get_camera_stream():
        ...     # Process frame here
        ...     cv2.imshow('Camera', frame)
        ...     if cv2.waitKey(1) & 0xFF == ord('q'):
        ...         break
        >>> cv2.destroyAllWindows()

        >>> # Custom resolution and fps
        >>> for frame in get_camera_stream(camera_index=0, width=640, height=480, fps=30):
        ...     # Process frame here
        ...     pass
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with index {camera_index}")

    try:
        # Set camera properties if specified
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            cap.set(cv2.CAP_PROP_FPS, fps)

        # Get actual camera properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Warning: Failed to read frame from camera")
                break

            yield frame

    finally:
        # Always release the camera
        cap.release()


def capture_single_frame(
    camera_index: int = 0, width: Optional[int] = None, height: Optional[int] = None
) -> np.ndarray:
    """
    Capture a single frame from the camera.

    Args:
        camera_index (int): Index of the camera device (default: 0)
        width (Optional[int]): Desired frame width in pixels
        height (Optional[int]): Desired frame height in pixels

    Returns:
        np.ndarray: Single camera frame as BGR image array

    Raises:
        RuntimeError: If camera cannot be opened or frame cannot be captured

    Example:
        >>> from facetoy.io.camera import capture_single_frame
        >>> frame = capture_single_frame()
        >>> cv2.imwrite('snapshot.jpg', frame)
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with index {camera_index}")

    try:
        # Set camera properties if specified
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Capture frame
        ret, frame = cap.read()

        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        return frame

    finally:
        cap.release()


def get_available_cameras() -> List[int]:
    """
    Get a list of available camera indices.

    Returns:
        list[int]: List of available camera indices

    Example:
        >>> from facetoy.io.camera import get_available_cameras
        >>> cameras = get_available_cameras()
        >>> print(f"Available cameras: {cameras}")
    """
    available_cameras = []

    # Test camera indices 0-10 (usually sufficient)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    return available_cameras
