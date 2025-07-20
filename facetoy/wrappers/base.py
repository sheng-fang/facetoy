from typing import List, Optional, Tuple

from pydantic import BaseModel


class FacePrediction(BaseModel):
    bbox_xywh: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: Optional[List[Tuple[int, int]]]  # List of (x, y) tuples for landmarks
    confidence: Optional[float] = None  # Confidence score, optional

    @property
    def x(self) -> int:
        return self.bbox_xywh[0]

    @property
    def y(self) -> int:
        return self.bbox_xywh[1]

    @property
    def width(self) -> int:
        return self.bbox_xywh[2]

    @property
    def height(self) -> int:
        return self.bbox_xywh[3]
