#!/usr/bin/env python3
"""Run object detection on an image, directory, or video source."""

from __future__ import annotations

import argparse
from pathlib import Path


def run_inference(source: str, weights: str, conf: float, imgsz: int, project: Path, name: str) -> list:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install project dependencies first: pip install -r requirements.txt") from exc

    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    predictions = model.predict(
        source=source,
        project=str(project),
        name=name,
        conf=conf,
        imgsz=imgsz,
        save=True,
        verbose=False,
    )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/raw", help="Image, directory, or video source to inspect.")
    parser.add_argument("--weights", default="training_outputs/pipeline_defects/weights/best.pt", help="Path to model weights.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--project", type=Path, default=Path("inspection"), help="Where output predictions are saved.")
    parser.add_argument("--name", default="inference_run", help="Output folder name.")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}. Train a model first.")

    predictions = run_inference(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    print(f"Processed {len(predictions)} prediction batch(es). Results saved under {args.project / args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
