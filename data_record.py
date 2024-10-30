import cv2 as cv
import os
import time



def capture_hand_images(folder_path):
    try:
        # Initialize camera
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Couldn't open the camera")

        # Ensure the folder exists
        os.makedirs(folder_path, exist_ok=True)

        counter = 0
        last_save_time = 0
        saving = False

        print("Press 's' to start saving images, 'p' to stop and exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error in capturing video frame")

            frame = cv.flip(frame, 1)
            cv.imshow("Camera", frame)

            # Handle key press
            key = cv.waitKey(1) & 0xFF
            if key == ord('s'):
                saving = True
            elif key == ord('p'):
                print("Stopping and closing the program.")
                break

            # Save images at intervals
            if saving and (time.time() - last_save_time) > 0.1:
                filename = f"{os.path.basename(folder_path)}_{counter}.jpg"
                save_path = os.path.join(folder_path, filename)
                cv.imwrite(save_path, frame)
                counter += 1
                last_save_time = time.time()
                print(f"Image {counter} saved to {save_path.replace(os.sep, '/')}")

        print(f"{counter} images saved in {folder_path.replace(os.sep, '/')}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Release resources
        if cap:
            cap.release()
        cv.destroyAllWindows()


# Call the function
capture_hand_images("DATA/CallMe")
