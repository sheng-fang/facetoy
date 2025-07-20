from pathlib import Path

import cv2

from facetoy.config import default_inference_cfg, default_static_cfg
from facetoy.pipeline.pipe_face_transform import PipeFaceTransform

image_path = Path(__file__).parent / "data/movie.jpeg"


def test_load_pipe() -> None:
    pipe = PipeFaceTransform.load(default_static_cfg)
    assert pipe is not None


def test_inference() -> None:
    pipe = PipeFaceTransform.load(default_static_cfg)
    image = cv2.imread(str(image_path))
    assert image is not None, "Failed to load test image"
    default_inference_cfg.face_transform.funhouse_mirror_effect.use = True
    results = pipe([image], default_inference_cfg)
    assert len(results) > 0, "Failed to detect"
