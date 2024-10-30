from ultralytics import YOLO
import cv2 as cv
import os

def run_detection(conf=0.5, save_video=False, filename=None):
    """
    Run real-time object detection on webcam feed using YOLOv8 model.

    Parameters:
    - conf: Confidence threshold for detections (default: 0.5)
    - save_video: Flag to save output as a video file (default: False)
    - filename: Optional filename for saved video. If None, an incremented filename is generated.
    """
    # Load the model
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Handle filename for saving video
    if save_video:
        if filename is None:
            # Generate an auto-incremented filename if none provided
            count = 1
            while os.path.exists(f"VIDEOS/detection_result_{count}.mp4"):
                count += 1
            filename = f"VIDEOS/detection_result_{count}.mp4"
        else:
            # Ensure .mp4 extension if user specifies filename
            if not filename.endswith(".mp4"):
                filename += ".mp4"
            filename = f"VIDEOS/{filename}"

        print(f"Recording video to {filename}")

    # Start video capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: couldn't open the webcam!")
        return

    # Capture frame width, height, and set fps
    w, h = (int(cap.get(x)) for x in (3, 4))
    fps = 15
    out = None

    # Set up video writer if saving is enabled
    if save_video:
        out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print("Press 's' to start saving, 'p' to pause and exit.")

    # Main loop to read frames and perform detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: couldn't read a frame from the webcam.")
            break

        # Flip the frame horizontally
        frame = cv.flip(frame, 1)

        # Run YOLO model prediction
        try:
            result = model.predict(frame, conf=conf, verbose=False)[0]
            frame = result.plot()
        except Exception as e:
            print(f"Error during model prediction: {e}")
            break

        # Display the frame
        cv.imshow("Camera", frame)

        # Handle user input
        key = cv.waitKey(1) & 0xFF
        if key == ord('s') and save_video:
            print("Video recording started.")
        elif key == ord('p'):
            print("Stopping and closing the program.")
            break

        # Save frame if enabled
        if save_video and out is not None:
            out.write(frame)

    # Release resources
    cap.release()
    if out:
        out.release()
    cv.destroyAllWindows()
    print("Program ended and resources released.")


