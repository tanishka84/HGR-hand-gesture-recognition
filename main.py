import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

class HandGestureDetector:
    """
    Real-time static hand gesture recognition using MediaPipe.
    Recognizes: Open Palm, Fist, Peace Sign, Thumbs Up, Ok, Thumbs Down (Dislike),
    Call Me, Rock On.
    """

    def __init__(self,
                 max_hands: int = 1,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def count_fingers(self, lm_list):
        fingers = 0
        if lm_list[4][0] > lm_list[3][0]:
            fingers += 1
        for i in range(1, 5):
            if lm_list[self.tip_ids[i]][1] < lm_list[self.tip_ids[i]-2][1]:
                fingers += 1
        return fingers

    def detect_gesture(self, lm_list, handedness):
        cnt = self.count_fingers(lm_list)

        if cnt == 0:
            return "Fist"
        if cnt == 5:
            return "Open Palm"
        if cnt == 2 and lm_list[8][1] < lm_list[6][1] and lm_list[12][1] < lm_list[10][1]:
            return "Peace Sign"
        if cnt == 2 and lm_list[8][1] < lm_list[6][1] and lm_list[20][1] < lm_list[18][1]:
            return "Rock On"
        if cnt == 2 and lm_list[4][0] > lm_list[3][0] and lm_list[20][1] < lm_list[18][1]:
            if all(lm_list[i][1] > lm_list[i-2][1] for i in [8,12,16]):
                return "Call Me"
        # Thumbs Up / Good Job
        if cnt == 1 and lm_list[4][1] < lm_list[3][1]:
            return "Good Job"
        # Ok: thumb and index touching, other three curled
        dist = np.hypot(lm_list[4][0]-lm_list[8][0], lm_list[4][1]-lm_list[8][1])
        if dist < 0.04 and \
           all(lm_list[i][1] > lm_list[i-2][1] for i in [12,16,20]):
            return "Ok"
        # Dislike (Thumbs Down)
        if lm_list[4][1] > lm_list[2][1] and all(lm_list[i][1] > lm_list[i-2][1] for i in [8,12,16,20]):
            return "Dislike"
        return "Unknown"

    def process_frame(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(img_rgb)
        gesture = None
        if res.multi_hand_landmarks:
            h, w, _ = frame.shape
            for lm, label in zip(res.multi_hand_landmarks, res.multi_handedness):
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
                lm_list = [(l.x, l.y) for l in lm.landmark]
                gesture = self.detect_gesture(lm_list, label.classification[0].label)
                cx, cy = int(lm_list[9][0]*w), int(lm_list[9][1]*h)
                # Display real-time gesture recognition text prominently
                cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            # When no hand detected, display text
            cv2.putText(frame, "Gesture: None", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        return frame, gesture

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Create directory structure - FIXED: Use separate variables for each directory
    # Get the project root directory (assuming the script runs from the project root)
    project_root = os.getcwd()  # Or define an absolute path like: project_root = "/path/to/project"

    # Define directories relative to the project root
    main_app_dir = os.path.join(project_root, "output-hand-gestures")
    videos_dir = os.path.join(main_app_dir, "VIDEOS")
    snapshots_dir = os.path.join(main_app_dir, "snapshots")
    logs_dir = os.path.join(main_app_dir, "logs")

    # Create directories (if they don't exist)
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(snapshots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Create all directories
    directories = [videos_dir, snapshots_dir, logs_dir]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Get camera properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"📹 Camera: {w}x{h} @ {actual_fps} FPS")

    # Video output setup - FIXED: Save in VIDEOS directory, not logs
    video_output_path = os.path.join(videos_dir, "output_gestures.mp4")
    
    # Multiple codec fallback for reliability
    codecs_to_try = [
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),
        ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('H264', cv2.VideoWriter_fourcc(*'H264'))
    ]
    
    out = None
    for codec_name, fourcc in codecs_to_try:
        out = cv2.VideoWriter(video_output_path, fourcc, 20.0, (w, h))
        if out and out.isOpened():
            print(f"✅ Video writer: {codec_name} codec")
            break
        else:
            if out:
                out.release()
            print(f"❌ Failed: {codec_name} codec")
    
    if not out or not out.isOpened():
        print("❌ ERROR: Could not initialize video writer!")
        cap.release()
        return

    # CSV file setup - FIXED: Save in logs directory
    csv_path = os.path.join(logs_dir, "gesture_log.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp", "datetime", "gesture", "confidence", "snapshot_path"])

    # Initialize detector
    detector = HandGestureDetector(max_hands=1, detection_confidence=0.7, tracking_confidence=0.5)
    prev_gesture = None
    prev_time = time.time()
    frame_count = 0
    gesture_count = 0

    print("\n🚀 Hand Gesture Recognition Started!")
    print("📹 Press 'q' to quit")
    print("📸 Press 's' for manual snapshot")
    print("-" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                break
                
            frame = cv2.flip(frame, 1)
            frame_count += 1

            # Process frame
            try:
                annotated, gesture_result = detector.process_frame(frame)
                
                # Handle gesture result
                if isinstance(gesture_result, tuple):
                    gesture, confidence = gesture_result
                else:
                    gesture = gesture_result
                    confidence = 0.9 if gesture and gesture != "Unknown" else 0.0
                    
            except Exception as e:
                print(f"❌ Detection error: {e}")
                annotated = frame.copy()
                gesture = None
                confidence = 0.0

            # Write to video
            success = out.write(annotated)
            if not success and frame_count % 30 == 0:
                print(f"⚠️ Video write failed at frame {frame_count}")

            # Handle gesture detection and snapshot
            if gesture and gesture != "Unknown" and gesture != prev_gesture:
                from datetime import datetime
                
                timestamp = time.time()
                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save snapshot - FIXED: Use correct snapshots directory
                clean_gesture = gesture.replace(" ", "_").lower()
                filename = f"{clean_gesture}_{int(timestamp)}.jpg"
                filepath = os.path.join(snapshots_dir, filename)
                
                # Save with high quality
                save_success = cv2.imwrite(filepath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                if save_success:
                    print(f"📸 Snapshot: {filename}")
                    gesture_count += 1
                    
                    # Log with full details
                    writer.writerow([timestamp, datetime_str, gesture, confidence, filepath])
                    csv_file.flush()
                    
                    prev_gesture = gesture
                    print(f"🤲 Gesture: {gesture} (confidence: {confidence:.2f})")
                else:
                    print(f"❌ Failed to save: {filename}")

            # Calculate and display FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            # Enhanced display
            cv2.putText(annotated, f"FPS: {int(fps)}", (10, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated, f"Frame: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated, f"Gestures: {gesture_count}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", annotated)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit command received")
                break
            elif key == ord('s'):
                # Manual snapshot
                manual_timestamp = int(time.time())
                manual_filename = f"manual_snapshot_{manual_timestamp}.jpg"
                manual_path = os.path.join(snapshots_dir, manual_filename)
                
                if cv2.imwrite(manual_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    print(f"📸 Manual snapshot: {manual_filename}")

    except KeyboardInterrupt:
        print("\n🛑 Program interrupted")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        cap.release()
        if out:
            out.release()
        csv_file.close()
        cv2.destroyAllWindows()
        
        # Verify outputs
        if os.path.exists(video_output_path):
            file_size = os.path.getsize(video_output_path)
            if file_size > 0:
                print(f"✅ Video saved: {video_output_path}")
                print(f"📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            else:
                print(f"⚠️ Video file is empty: {video_output_path}")
        else:
            print(f"❌ Video not created: {video_output_path}")
        
        print(f"\n📊 Session Statistics:")
        print(f"   • Frames processed: {frame_count}")
        print(f"   • Gestures detected: {gesture_count}")
        print(f"   • Video: {video_output_path}")
        print(f"   • Snapshots: {snapshots_dir}")
        print(f"   • Log: {csv_path}")
        print("🏁 Program terminated cleanly.")


if __name__ == "__main__":
    main()