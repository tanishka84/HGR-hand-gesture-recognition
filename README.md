# Hand Gesture Detection using YOLOv8

This repository features a project for **Hand Gesture Detection** utilizing Python, OpenCV, and YOLOv8. The project allows for real-time data recording, annotation, and model development aimed at recognizing specific hand gestures.

![detection_result_2](https://github.com/user-attachments/assets/0acb942d-edf2-4f0b-847f-b80f3c78330f)

## Features

- **Hand Gesture Detection**:
  - Captures hand gesture images from a webcam feed.
  - Images are saved in specified folders for each gesture.
  - Key-activated controls to start and stop image saving.
- **Dataset Preparation**:
  - Provides a script to record hand gesture data.
  - Supports annotation of the recorded data using Roboflow.
- **Model Training and Deployment**:
  - Includes a Jupyter Notebook for training, evaluating, and deploying the YOLOv8 hand gesture detection model.
- **Separate Python Script real-time detection**:
  - Includes a separate Python script that detects and displays hand gestures from a live camera feed.

## Project Structure

- `data_record.py`: Python script to record webcam feed data for hand gesture detection.
- `model_training.ipynb`: Jupyter Notebook for training, testing, and deploying the hand gesture detection model.
- `requirements.txt`: List of required Python packages.
- `testing.py`: Python script for real-time hand gesture detection.

## Setup and Requirements

### Prerequisites

- Python 3.x
- OpenCV
- YOLOv8
- Roboflow (for data annotation)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Pushtogithub23/yolo-hand-gestures-detection.git
   cd yolo-hand-gestures-detection
   ```

2. Install required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Train your custom YOLOv8 model weights. Typically, the weights are saved in `runs/detect/train/weights/`.

## Usage

### Hand Gesture Data Recording

The `data_record.py` script captures images from a webcam feed to create a dataset of hand gestures.

1. Run the script:

   ```bash
   python data_record.py
   ```

2. **Controls**:
   - Press **'s'** to start saving images.
   - Press **'p'** to stop and close the program.

   This will save images in the `DATA/CallMe` directory or any folder specified in `capture_hand_images()`.

### Data Annotation

After recording the hand gesture data, images were annotated on [Roboflow](https://roboflow.com/). Annotation was essential for training the YOLOv8 model with labelled gesture data.

### Hand Gesture Detection Notebook

The `model_training.ipynb` notebook provides:

- Training and testing steps for the hand gesture detection model.
- Code to evaluate model performance.
- Guidance for deploying the trained model.

## Project link
 - You can find the project on Roboflow by clicking [here](https://universe.roboflow.com/puspendu-ai-vision-workspace/hand_gesture_detection-xdcpy)
 - You can find the training logs on wandb by clicking on this [link](https://wandb.ai/ranapuspendu24-iit-madras-foundation/Ultralytics/runs/zayjstbb?nw=nwuserranapuspendu24)

