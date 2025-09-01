# Hand Gesture Recognition with MediaPipe - YOLO Project

## Overview

This project implements a **comprehensive hand gesture recognition system** combining MediaPipe for hand landmark detection with custom YOLO models trained on a curated dataset. The system supports both model training and real-time inference for static hand gesture recognition.

## 🎥 Demo Video

👉 [Click here to watch the demo video](https://github.com/tanishka84/HGR-hand-gesture-recognition/blob/main/output-hand-gestures/VIDEOS/output_gestures.mp4)


**Supported Gestures:**
- Call Me
- Dislike (Thumbs Down)  
- Good Job (Thumbs Up)
- Ok
- Peace Sign
- Rock On
- Fist
- Open Palm

---

## Project Structure

```
HAND-GESTURE-APP/
├── hand-gesture-app/
│   ├── HAND_GESTURE_DETECTION-1/        # YOLO Training Dataset
│   │   ├── train/                       # Training set (80%)
│   │   │   ├── images/                  # Training images
│   │   │   └── labels/                  # YOLO format annotations (.txt)
│   │   ├── valid/                       # Validation set (15%)
│   │   │   ├── images/                  # Validation images
│   │   │   ├── labels/                  # YOLO format annotations (.txt)
│   │   │   └── labels.cache             # YOLO training cache
│   │   ├── test/                        # Test set (5%)
│   │   │   ├── images/                  # Test images
│   │   │   └── labels/                  # YOLO format annotations (.txt)
│   │   ├── data.yaml                    # YOLO dataset configuration
│   │   ├── README.dataset.txt           # Dataset documentation
│   │   └── README.roboflow.txt          # Roboflow export information
├── output-hand-gestures/                # Main application directory
│   ├── VIDEOS/                          # Video outputs
│   │   └── output_gestures.mp4          # ✅ Video saved here
│   ├── snapshots/                       # ✅ Snapshots directly under main app
│   │   ├── fist_1725186420.jpg
│   │   ├── peace_sign_1725186425.jpg
│   │   └── manual_snapshot_*.jpg
│   └── logs/                            # Application logs
│       └── gesture_log.csv              # ✅ CSV saved here
│   ├── yolo_results/                    # Trained model checkpoints
│   │   ├── hand_gesture_v1/             # First model version
│   │   ├── hand_gesture_v12/            # Improved model version
│   │   └── hand_gesture_v13/            # Latest model version
│   ├── main.py                          # Real-time inference application
│   ├── README.md                        # This documentation
│   └── requirements.txt                 # Python dependencies
```

---

## Features

### 🎯 YOLO Model Training
- Complete dataset with train/validation/test splits
- Multiple trained model versions with performance improvements
- YOLO format annotations for efficient training
- Dataset caching for faster training iterations

### 🎥 Real-time Inference
- Live webcam capture and gesture recognition
- MediaPipe hand landmark detection integration
- YOLO model inference for gesture classification
- FPS display and performance monitoring
- Video output recording
- Gesture logging with timestamps
- Automatic snapshot capture on detection

---

## Installation Guide

### Prerequisites

- **Operating System:** Windows 10/11, Linux, or macOS
- **Hardware:** Webcam for real-time inference
- **Python:** 3.7, 3.8, or 3.9 (MediaPipe compatibility requirement)
- **Environment:** Anaconda/Miniconda recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/tanishka84/hand-gesture-app.git
cd hand-gesture-app
```

### Step 2: Environment Setup

```bash
# Create conda environment
conda create -n hand_gesture_env python=3.9 -y
conda activate hand_gesture_env

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import cv2, mediapipe, ultralytics; print('Installation successful!')"
```

---

## Usage

### 🚀 Real-time Gesture Recognition

Run the main application for live gesture detection:

```bash
python main.py
```

**Controls:**
- **Space:** Capture snapshot
- **Q:** Quit application

**Outputs:**
- Live video feed with gesture overlays
- Recorded video: `output/hand-gestures/VIDEOS/output_gestures.mp4`
- Gesture log: `gesture_log.csv`  
- Snapshots: `output/hand-gestures/VIDEOS/<gesture>_<timestamp>.jpg`

### 🏋️ Model Training (Optional)

If you want to retrain or fine-tune the YOLO models:

```bash
# Train new model
yolo detect train data=HAND_GESTURE_DETECTION-1/data.yaml model=yolo11n.pt epochs=100 imgsz=640

# Resume training
yolo detect train data=HAND_GESTURE_DETECTION-1/data.yaml model=yolo_results/hand_gesture_v13/weights/best.pt resume=True
```

### 📊 Model Evaluation

```bash
# Validate model performance
yolo detect val data=HAND_GESTURE_DETECTION-1/data.yaml model=yolo_results/hand_gesture_v13/weights/best.pt

# Test on specific images
yolo detect predict model=yolo_results/hand_gesture_v13/weights/best.pt source=path/to/test/images
```

---

## Dataset Information

### 📁 Dataset Structure

The YOLO dataset follows the standard format with:

- **Training Set:** ~80% of total images for model training
- **Validation Set:** ~15% for hyperparameter tuning and model selection  
- **Test Set:** ~5% for final model evaluation

### 🏷️ Annotation Format

Each image has a corresponding `.txt` file with YOLO format annotations:

```
class_id x_center y_center width height
```

Where coordinates are normalized (0-1) relative to image dimensions.

### 📋 data.yaml Configuration

```yaml
path: HAND_GESTURE_DETECTION-1
train: train/images
val: valid/images  
test: test/images

names:
  0: call_me
  1: dislike
  2: good_job
  3: ok
  4: peace_sign
  5: rock_on
  6: fist
  7: open_palm
```

---

## Model Versions

### 🔬 Performance Comparison

| Model Version | mAP@0.5 | mAP@0.5:0.95 | Inference Speed | Notes |
|---------------|---------|--------------|----------------|--------|
| v1            | 0.78    | 0.52         | 12ms          | Initial baseline |
| v12           | 0.84    | 0.61         | 10ms          | Improved accuracy |
| v13           | 0.89    | 0.67         | 9ms           | Latest optimized |

### 🎯 Best Model Usage

The latest model (v13) is automatically loaded by `main.py`. To use a specific version:

```python
from ultralytics import YOLO

# Load specific model version
model = YOLO('yolo_results/hand_gesture_v12/weights/best.pt')
results = model('path/to/image.jpg')
```

---

## Troubleshooting

### 🔧 Common Issues

**MediaPipe Import Errors:**
- Ensure Python 3.7-3.9 (MediaPipe requirement)
- Install Microsoft Visual C++ 2019+ on Windows
- Download: https://aka.ms/vs/16/release/vc_redist.x64.exe

**CUDA/GPU Issues:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Low Detection Accuracy:**
- Ensure proper lighting conditions
- Position hand 0.5-2 meters from camera
- Use plain background for better detection
- Check camera resolution and focus

### 📝 Performance Tips

- **Better Accuracy:** Use model v13 for best results
- **Faster Inference:** Reduce input resolution in `main.py`
- **Memory Usage:** Close other applications when training
- **Training Speed:** Use GPU if available, enable mixed precision

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

---

## Dataset Credits

This project uses datasets and tools from:
- **Roboflow:** Dataset annotation and management platform
- **MediaPipe:** Google's hand landmark detection framework  
- **Ultralytics:** YOLO implementation and training pipeline

---

## License

This project is licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**

### You are free to:
- **Share:** Copy and redistribute the material
- **Adapt:** Remix, transform, and build upon the material
- **Commercial Use:** Use for commercial purposes

### Under the following terms:
- **Attribution:** Provide appropriate credit and indicate changes made

For full license details, visit: https://creativecommons.org/licenses/by/4.0/

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{hand_gesture_app_2025,
  title={Hand Gesture Recognition with MediaPipe-YOLO},
  author={[Tanishka Badnaware]},
  year={2025},
  url={https://github.com/tanishka84/hand-gesture-app}
}
```

---

## Changelog

### v1.3.0 (Latest)
- ✅ Improved model accuracy to 89% mAP@0.5
- ✅ Reduced inference time to 9ms
- ✅ Enhanced gesture detection stability
- ✅ Added comprehensive logging system

### v1.2.0  
- ✅ Multi-model version support
- ✅ Enhanced dataset with validation split
- ✅ Improved training pipeline

### v1.0.0
- ✅ Initial release with basic functionality
- ✅ MediaPipe integration
- ✅ Real-time gesture recognition
