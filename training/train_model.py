#!/usr/bin/env python3
"""Train a YOLOv8 model for pipeline defect detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def train_model(dataset_yaml: Path, weights: str, epochs: int, imgsz: int, batch: int, project: Path, name: str) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install project dependencies first: pip install -r requirements.txt") from exc

    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        pretrained=True,
        workers=4,
        verbose=True,
    )
    best_path = project / name / "weights" / "best.pt"
    return best_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/data.yaml"), help="YOLO dataset YAML from the prepared dataset.")
    parser.add_argument("--weights", default="yolov8n.pt", help="Base model weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for training.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--project", type=Path, default=Path("training_outputs"), help="Directory to save training artifacts.")
    parser.add_argument("--name", default="pipeline_defects", help="Run name for the model output.")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset YAML not found: {args.dataset}. Run training/prepare_dataset.py first.")

    best_path = train_model(
        dataset_yaml=args.dataset,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(f"Training completed. Best weights saved to {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
