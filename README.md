# RTMaskRCNN - Real-Time Instance Segmentation with Principal Axes

This notebook demonstrates real-time object detection and instance segmentation using Mask R-CNN (with a ResNet-50 FPN backbone) on webcam input. For each detected object, it visualizes the bounding box, class label, segmentation mask, and computes the principal axes and centroid of the mask using PCA.

## Features

- Real-time webcam inference using a pre-trained Mask R-CNN model from PyTorch.
- Visualization of detected objects with bounding boxes, class labels, and colored masks.
- Calculation and drawing of the centroid and principal axes for each detected mask.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (`cv2`)
- scikit-learn
- numpy
- PIL (Pillow)
- matplotlib

Install dependencies with:
```bash
pip install torch torchvision opencv-python scikit-learn numpy pillow matplotlib
```

## Usage

1. Ensure your webcam is connected.
2. Run all cells in this notebook.
3. The webcam window will display real-time detections. Press `q` to quit.

## Notes

- The model uses COCO dataset labels.
- Confidence threshold for detections is set to 0.65.
- Principal axes are computed using PCA on the mask pixels.
