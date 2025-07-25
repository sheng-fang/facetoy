"""Example usage of the FaceDetectorOpenCV class."""

from pathlib import Path

import cv2

from facetoy.io import get_camera_stream
from facetoy.utils.face_transform import funhouse_mirror_effect
from facetoy.utils.vis import plot_rectangles
from facetoy.wrappers import FaceDetectorOpenCV

cv2_path = Path(cv2.__file__).parent


def main() -> None:
    """Demonstrate face detection with camera stream."""
    print("Initializing face detector...")

    # Initialize face detector with default Haar cascade
    face_detector = FaceDetectorOpenCV(
        model_type="haar_frontalface",
    )

    print(f"Face detector info: {face_detector.get_model_info()}")
    print("Starting camera stream with face detection...")
    print("Press 'q' to quit, 's' to save current frame with detections")

    frame_count = 0

    try:
        # Get camera stream
        for frame in get_camera_stream(camera_index=0, width=640, height=480):
            frame_count += 1

            # Detect faces in the current frame
            face_predictions = face_detector.forward(frame)
            for prediction in face_predictions:
                x, y, w, h = prediction.bbox_xywh
                face_image = frame[y : y + h, x : x + w]
                face_transformed = funhouse_mirror_effect(face_image, distortion_strength=0.5)
                frame[y : y + h, x : x + w] = face_transformed

            # Draw bounding boxes around detected faces
            frame = plot_rectangles(
                frame, [p.bbox_xywh for p in face_predictions], color=(0, 255, 0), thickness=2
            )  # Green color

            # Add information overlay
            info_text = f"Frame: {frame_count} | Faces: {len(face_predictions)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the frame
            cv2.imshow("Face Detection Demo", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                filename = f"face_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")


def test_single_image() -> None:
    """Test face detection on a single image file."""
    # You can test with your own image file
    image_path = "tests/data/movie.jpeg"

    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return

        # Initialize face detector
        face_detector = FaceDetectorOpenCV(model_type="haar_frontalface_default")

        # Detect faces
        face_predictions = face_detector.forward(image)
        print(f"Detected {len(face_predictions)} faces")

        # Draw bounding boxes
        result_image = plot_rectangles(image, [p.bbox_xywh for p in face_predictions])

        # Save result
        print("Result saved as face_detection_result.jpg")

        # Display result
        cv2.imshow("Face Detection Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    # Choose which demo to run
    print("Face Detection Demo Options:")
    print("1. Real-time camera detection")
    print("2. Single image detection")

    choice = input("Enter your choice (1-2): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        test_single_image()
    else:
        print("Invalid choice, running real-time detection...")
        main()
