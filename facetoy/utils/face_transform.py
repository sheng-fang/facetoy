from typing import Optional

import numpy as np


def funhouse_mirror_effect(
    image: np.ndarray[np.uint8],
    distortion_strength: float = 0.8,
    center_x: Optional[int] = None,
    center_y: Optional[int] = None,
    radius: Optional[int] = None,
) -> np.ndarray[np.uint8]:
    height, width, _ = image.shape
    output_img = np.zeros_like(image)
    center_x = center_x if center_x is not None else width // 2
    center_y = center_y if center_y is not None else height // 2
    radius = radius if radius is not None else min(width, height) // 2

    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < radius:
                scale = 1 - (distortion_strength * (1 - distance / radius))

                original_x = int(center_x + dx * scale)
                original_y = int(center_y + dy * scale)

                if 0 <= original_x < width and 0 <= original_y < height:
                    output_img[y, x] = image[original_y, original_x]
                else:
                    output_img[y, x] = [0, 0, 0]  # 黑色
            else:
                output_img[y, x] = image[y, x]

    return output_img
