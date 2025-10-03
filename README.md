# Tree-Species-Detection with YOLOv8
Tree species detection with YOLOv8 automates identifying and locating tree types in images or video for forest management, biodiversity, ecology, and precision agriculture—and in this project, a Raspberry Pi was used for real-time prototyping.

---

## Features

- Deep learning model training with YOLOv8
- Real-time tree species detection on a Raspberry Pi
- Live video stream annotation
- Custom model evaluation on sample images
  
---

## Requirements

- Python 3.9 or newer
- pip (Python package installer)

Install all dependencies with:
pip install ultralytics opencv-python torch matplotlib pyyaml picamera2

*`picamera2` is only required for running real-time detection on Raspberry Pi.*

---

## Usage

### 1. Training the Model (`YOLO.ipynb`)

- Prepare your dataset in YOLO or COCO format.
- Edit the `DATAYAMLPATH` and other parameters as needed in `YOLO.ipynb`.
- Run the notebook with:
  jupyter notebook YOLO.ipynb
- Train the model. The final model weights (e.g., `best.pt`) will be saved at the end.

#### **Testing Custom Models and Images**
Users can also test their own trained models in the notebook by changing the model path and image source, for example:

from ultralytics import YOLO

model = YOLO("runs/detect/train6/weights/bestV2.pt")
results = model.predict(source='Dataset/Cannonball/can_32.jpg', save=False, conf=0.4)

Adjust paths to any model file and image to quickly visualize detection outputs with bounding boxes and confidence scores.

### 2. Real-Time Detection (`tree-species.py`)

- Copy your trained weights file (e.g., `bestV2.pt`) to your Raspberry Pi.
- Ensure the Pi camera is connected and enabled.
- Run:
  python tree-species.py
- The script will display a window with live detection results. Press `q` to exit.

---

## Notes

- For GPU training, ensure CUDA drivers are installed and compatible with your torch version.
- The model weights used for detection must match the class set used in training.
- For Raspberry Pi, use Raspberry Pi OS and ensure your camera works with `picamera2`.


  
