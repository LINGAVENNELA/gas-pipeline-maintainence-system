# Gas Pipeline Maintenance System

AI-powered gas pipeline inspection system for autonomous defect detection and maintenance monitoring. The project combines a YOLO object-detection pipeline with dataset validation, model training, live inference, and rover inspection workflows for identifying defects like deformation, rupture, misalignment, disconnects, obstacles, and deposition in pipeline infrastructure.

## What this project includes

- Dataset download integration for the baseline pipeline defect dataset
- Dataset auditing and validation for YOLO annotation quality
- Dataset preparation into a train/validation split for YOLOv8
- Model training workflow for defect detection
- Inference pipeline for images, directories, and video sources
- Live webcam inspection loop for on-robot defect monitoring
- Project structure ready for Raspberry Pi and camera-based robotic deployment

## Repository layout

- [scripts/download_dataset.py](scripts/download_dataset.py): downloads the source dataset into data/raw
- [training/dataset_audit.py](training/dataset_audit.py): validates dataset labels, dimensions, classes, and annotation health
- [training/prepare_dataset.py](training/prepare_dataset.py): converts the raw dataset into YOLO train/validation folders and writes data.yaml
- [training/train_model.py](training/train_model.py): trains a YOLOv8 model on the prepared dataset
- [inference/run_inference.py](inference/run_inference.py): runs inference against images or videos
- [rover/inspection_loop.py](rover/inspection_loop.py): live camera inspection loop with alerting logic
- [requirements.txt](requirements.txt): Python dependencies for training and inference

## Getting started

1. Create and activate a Python environment.
2. Install dependencies:

   pip install -r requirements.txt

3. Download the dataset:

   python scripts/download_dataset.py

4. Audit the dataset:

   python training/dataset_audit.py --dataset data/raw

5. Prepare the dataset for training:

   python training/prepare_dataset.py --dataset data/raw --output data/processed

6. Train the defect detection model:

   python training/train_model.py --dataset data/processed/data.yaml --epochs 50 --imgsz 640

7. Run inference on an image or directory:

   python inference/run_inference.py --source data/raw --weights training_outputs/pipeline_defects/weights/best.pt

8. Launch the live rover inspection loop:

   python rover/inspection_loop.py --weights training_outputs/pipeline_defects/weights/best.pt --camera-index 0

## Expected defect classes

The baseline project uses the following labels:

- 0: Deformation
- 1: Obstacle
- 2: Rupture
- 3: Disconnect
- 4: Misalignment
- 5: Deposition

## Outputs

- Dataset reports are written to the reports directory.
- Prepared training data is saved under data/processed.
- Model checkpoints are stored under training_outputs.
- Inference results are stored under inspection.

## Notes

- This repository is designed for practical pipeline inspection work and can be extended to Raspberry Pi hardware integration, trained model deployment, alerting, and autonomous navigation.
- The live rover workflow is structured for camera-based inspection and can be connected to GPIO-controlled movement and sensor modules as hardware is added.
