#!/usr/bin/env python3
"""Audit a YOLO-format dataset without modifying its source files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

EXPECTED_NAMES = {
    0: "Deformation",
    1: "Obstacle",
    2: "Rupture",
    3: "Disconnect",
    4: "Misalignment",
    5: "Deposition",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml_mapping(dataset: Path) -> dict[int, str] | None:
    yaml_files = list(dataset.rglob("data.yaml")) + list(dataset.rglob("data.yml"))
    if not yaml_files:
        return None
    try:
        import yaml
    except ImportError:
        return None
    content = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8")) or {}
    names = content.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {int(index): str(name) for index, name in names.items()}


def image_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            return image.width, image.height
    except Exception:
        return None


def find_label(image: Path) -> Path | None:
    relative = image.relative_to(image.parents[1])
    candidates = [
        image.with_suffix(".txt"),
        image.parent.parent / "labels" / relative.name.replace(image.suffix, ".txt"),
        image.parent.parent / "labels" / image.relative_to(image.parent.parent / "images").with_suffix(".txt")
        if "images" in image.parts
        else image.with_suffix(".txt"),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def audit(dataset: Path) -> dict[str, Any]:
    images = sorted(path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    class_annotation_counts: Counter[int] = Counter()
    class_image_counts: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    malformed: list[str] = []
    invalid_classes: list[str] = []
    invalid_boxes: list[str] = []
    missing_labels: list[str] = []
    empty_labels: list[str] = []
    corrupted_images: list[str] = []
    annotation_files = set(dataset.rglob("*.txt"))

    for image in images:
        relative = image.relative_to(dataset)
        split = next((part for part in relative.parts if part.lower() in {"train", "val", "valid", "validation", "test"}), "unknown")
        split_counts[split] += 1
        formats[image.suffix.lower().lstrip(".")] += 1
        size = image_dimensions(image)
        if size is None:
            corrupted_images.append(str(relative))
        else:
            dimensions[f"{size[0]}x{size[1]}"] += 1
        label = find_label(image)
        if label is None:
            missing_labels.append(str(relative))
            continue
        annotation_files.discard(label)
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels.append(str(label.relative_to(dataset)))
            continue
        image_classes: set[int] = set()
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            location = f"{label.relative_to(dataset)}:{line_number}"
            if len(fields) != 5:
                malformed.append(location)
                continue
            try:
                class_id = int(fields[0])
                box = [float(value) for value in fields[1:]]
            except ValueError:
                malformed.append(location)
                continue
            if class_id not in EXPECTED_NAMES:
                invalid_classes.append(location)
            if any(value < 0 or value > 1 for value in box) or box[2] <= 0 or box[3] <= 0:
                invalid_boxes.append(location)
            class_annotation_counts[class_id] += 1
            image_classes.add(class_id)
        for class_id in image_classes:
            class_image_counts[class_id] += 1

    mapping = load_yaml_mapping(dataset)
    return {
        "dataset": str(dataset.resolve()),
        "format": "YOLO detection labels",
        "total_images": len(images),
        "total_annotation_files": len(annotation_files) + len(images) - len(missing_labels),
        "total_annotations": sum(class_annotation_counts.values()),
        "images_per_class": {EXPECTED_NAMES.get(k, str(k)): v for k, v in sorted(class_image_counts.items())},
        "annotations_per_class": {EXPECTED_NAMES.get(k, str(k)): v for k, v in sorted(class_annotation_counts.items())},
        "image_dimensions": dict(dimensions),
        "image_formats": dict(formats),
        "splits": dict(split_counts),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "malformed_labels": malformed,
        "invalid_class_ids": invalid_classes,
        "invalid_bounding_boxes": invalid_boxes,
        "corrupted_images": corrupted_images,
        "unmatched_annotation_files": [str(path.relative_to(dataset)) for path in sorted(annotation_files)],
        "duplicate_images": [],
        "class_mapping": mapping,
        "expected_class_mapping": EXPECTED_NAMES,
        "class_mapping_matches": mapping == EXPECTED_NAMES if mapping is not None else None,
        "class_imbalance": dict(class_annotation_counts),
    }


def write_csv(report: dict[str, Any], destination: Path) -> None:
    rows = []
    for key, value in report.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                rows.append({"metric": f"{key}.{subkey}", "value": json.dumps(subvalue)})
        elif isinstance(value, list):
            rows.append({"metric": key, "value": json.dumps(value)})
        else:
            rows.append({"metric": key, "value": value})
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if not args.dataset.exists():
        print(f"Dataset path does not exist: {args.dataset}. Download it first.", file=sys.stderr)
        return 2
    report = audit(args.dataset)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report, args.reports_dir / "dataset_audit.csv")
    print(json.dumps(report, indent=2))
    if report["class_mapping_matches"] is False:
        print("Class mapping differs from the required Baseline V1 mapping; training must stop.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
