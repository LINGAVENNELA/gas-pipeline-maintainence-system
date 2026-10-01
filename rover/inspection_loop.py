#!/usr/bin/env python3
"""Run live pipeline inspection using the camera and a trained defect-detection model."""

from __future__ import annotations

import argparse
from pathlib import Path


class SensorSuite:
    """Minimal hardware abstraction for future Raspberry Pi sensor integration."""

    def __init__(self, ultrasonic_trig: int | None = None, ultrasonic_echo: int | None = None):
        self.ultrasonic_trig = ultrasonic_trig
        self.ultrasonic_echo = ultrasonic_echo

    def read_distance_cm(self) -> float | None:
        return None

    def read_step_count(self) -> int:
        return 0


def live_inspection(weights: str, camera_index: int = 0, conf: float = 0.25, max_frames: int = 100) -> None:
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Install project dependencies first: pip install -r requirements.txt") from exc

    model = YOLO(weights)
    sensor_suite = SensorSuite()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    frame_count = 0
    while frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 5 == 0:
            results = model(frame, conf=conf, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0]]
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf_value = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf_value:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if label in {"Rupture", "Disconnect", "Deposition"}:
                    print(f"ALERT: {label} detected with confidence {conf_value:.2f}")

        cv2.imshow("Gas Pipeline Inspection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="training_outputs/pipeline_defects/weights/best.pt", help="Path to the trained model weights.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to inspect from.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum number of frames to inspect.")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}. Train a model first.")

    live_inspection(
        weights=args.weights,
        camera_index=args.camera_index,
        conf=args.conf,
        max_frames=args.max_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
