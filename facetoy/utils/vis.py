from typing import List, Tuple

import cv2
import numpy as np


def plot_rectangles(
    image: np.ndarray,
    rectangles: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    result_image = image.copy()

    for x, y, w, h in rectangles:
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)

    return result_image
