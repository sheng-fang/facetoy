"""Example usage of the get_camera_stream function."""

import cv2

from facetoy.io import get_camera_stream


def main() -> None:
    """Demonstrate camera streaming with the get_camera_stream function."""
    print("Starting camera stream...")
    print("Press 'q' to quit, 's' to save current frame")

    frame_count = 0

    try:
        # Get camera stream with custom resolution
        for frame in get_camera_stream(camera_index=0, width=640, height=480, fps=30):
            frame_count += 1

            # Add frame counter to the image
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow("Camera Stream", frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord("s"):
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

    except RuntimeError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_count}")


if __name__ == "__main__":
    main()
