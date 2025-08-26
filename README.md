## Vehicle License Plate Recognition (YOLO + EasyOCR + PyQt5)

A desktop app that detects license plate regions with YOLO and recognizes characters with EasyOCR, running on a PyQt5 GUI. The project includes training, image/camera inference, and data augmentation scripts.

### Table of Contents
- **Overview**
- **Requirements**
- **Prepare model (.pt)**
- **Project structure**
- **Run the GUI app**
- **Train the model**
- **Data organization and augmentation**
- **Rename dataset files**
- **Contributing & License**
- **Tips & Troubleshooting**

## Overview
- **Detection**: Ultralytics YOLO locates license plate bounding boxes on images/frames.
- **Recognition**: EasyOCR reads characters from the extracted and preprocessed plate region.
- **UI**: PyQt5 with two sample screens: image processing (`giaodien_image.ui`) and camera (`giaodien_camera.ui`).

## Requirements
- Python 3.8–3.11 (3.10+ recommended)
- Windows 10/11; CUDA GPU optional

Install dependencies:
```bash
pip install -r requirements.txt
```
Note: `ultralytics` depends on `torch`. If you hit Torch/CUDA issues, install PyTorch manually per their official guide.

## Project structure
```text
XLA_BSX/
  README.md
  requirements.txt
  .gitignore
  models/
    yolo11_trained_v2.pt
  ui/
    giaodien_image.ui
    giaodien_camera.ui
  data/
    raw/
      images/
      labels/
    processed/
      images/
      labels/
  src/
    test_image.py
    test_camera.py
    train.py
    augment/
      rotate.py
      Scale.py
      Translation.py
      noise.py
    tools/
      change_name.py
  configs/
    data.yaml.example
```
You can keep the previous layout, but the structure above is recommended for maintainability.

## Prepare model (.pt)
Place your YOLO weights (e.g., `yolo11_trained_v2.pt` or `best.pt`) in `models/`. Update paths in the inference scripts if needed:

In `test_image.py`:
```python
model = YOLO("models/yolo11_trained_v2.pt")  # or your .pt path
```

In `test_camera.py`:
```python
model = YOLO("models/yolo11_trained_v2.pt")
```

## Run the GUI app
### 1) Inference on a still image
```bash
python src/test_image.py
```
- Select an image, run detection + OCR, and visualize bounding boxes, confidence, and recognized text.

### 2) Inference from camera
```bash
python src/test_camera.py
```
- Toggle the camera; the live view shows detections and OCR results.
- If the default camera fails, try changing `cv2.VideoCapture(0)` to `1` or `2`.

## Train the model
`train.py` loads a YOLO model and calls `model.train(...)`. Update `configs/data.yaml` (copy from `configs/data.yaml.example`) to match your dataset structure:
```python
results = model.train(
    data="configs/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0,      # GPU 0; use -1 or omit if CPU-only
    workers=2,
)
```
Run training:
```bash
python src/train.py
```
The trained weights are saved to `models/yolo11_trained_v2.pt`.

## Data organization and augmentation
Default dataset folders in this project:
- `data/raw/images`, `data/raw/labels`: raw images and YOLO-format labels (`class x_center y_center width height`, normalized 0–1)
- `data/processed/images`, `data/processed/labels`: outputs from augmentation scripts

Augmentation scripts (run from the project root):
- `src/augment/rotate.py`: rotate images and update labels. Outputs to `data/processed/images` and `data/processed/labels`.
  ```bash
  python src/augment/rotate.py
  ```
- `src/augment/Scale.py`: scale images and adjust labels. Outputs to `data/processed/images` and `data/processed/labels`.
  ```bash
  python src/augment/Scale.py
  ```
- `src/augment/Translation.py`: translate images and adjust labels. Outputs to `data/processed/images` and `data/processed/labels`.
  ```bash
  python src/augment/Translation.py
  ```
- `src/augment/noise.py`: add salt-and-pepper noise. Outputs to `data/processed`.
  ```bash
  python src/augment/noise.py
  ```

Notes:
- Scripts sample a random subset from `data/raw/images`. Adjust the sample size as needed.
- Ensure `data/processed/images` and `data/processed/labels` exist before running.

## Rename dataset files
`src/tools/change_name.py` renames paired image/label files in `data/processed/images` and `data/processed/labels` to a sequential pattern (`image_1.jpg`, `image_1.txt`, ...).
```bash
python src/tools/change_name.py
```
Back up your data before renaming to avoid unintended overwrites.

## Contributing & License
- Contributions via Pull Requests are welcome. Please follow Conventional Commits and describe your changes clearly.
- License: MIT (or update per the LICENSE file if provided).

## Tips & Troubleshooting
- **Torch/CUDA**: install the correct Torch build for your CUDA/CPU environment per PyTorch docs.
- **EasyOCR without GPU**: change `easyocr.Reader(['en'], gpu=True)` to `gpu=False`.
- **Camera not opening**: check permissions; try device index `0 -> 1/2`.
- **Training data path**: update `configs/data.yaml` to fit your machine/dataset layout.

---
If you need guidance on integrating new data or packaging the app, please open an issue or reach out.
