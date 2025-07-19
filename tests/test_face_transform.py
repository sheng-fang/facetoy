import cv2

from facetoy.utils.face_transform import funhouse_mirror_effect

image_path = "/Users/sheng/Code/github.com/facetoy/tests/data/cartoon.jpg"  # Ensure this path is correct


def test_funhouse_mirror_effect() -> None:
    img_test = cv2.imread(image_path)
    try:
        funhouse_mirror_effect(img_test, distortion_strength=1.8)
    except Exception as e:
        assert False, f"Error occurred while applying funhouse mirror effect: {e}"

    assert True, "Funhouse mirror effect applied successfully without exceptions"
