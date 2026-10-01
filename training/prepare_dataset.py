#!/usr/bin/env python3
"""Prepare a YOLO dataset into a train/validation split for model training."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

EXPECTED_NAMES = {
    0: "Deformation",
    1: "Obstacle",
    2: "Rupture",
    3: "Disconnect",
    4: "Misalignment",
    5: "Deposition",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_image_files(dataset: Path) -> list[Path]:
    return sorted(
        path for path in dataset.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_label_path(image_path: Path, dataset: Path) -> Path | None:
    candidates = [
        image_path.with_suffix(".txt"),
        dataset / "labels" / image_path.name.replace(image_path.suffix, ".txt"),
    ]
    parent = image_path.parent
    if "images" in image_path.parts:
        relative = image_path.relative_to(dataset / "images")
        candidates.append(dataset / "labels" / relative.with_suffix(".txt"))
    if parent.parent.name == "images":
        candidates.append(parent.parent / "labels" / image_path.name.replace(image_path.suffix, ".txt"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def create_split(images: list[Path], dataset: Path, train_ratio: float = 0.8) -> tuple[list[Path], list[Path]]:
    if any(part.lower() in {"train", "val", "valid", "validation", "test"} for image in images for part in image.relative_to(dataset).parts):
        grouped: dict[str, list[Path]] = {"train": [], "val": []}
        for image in images:
            relative = image.relative_to(dataset)
            split = next((part for part in relative.parts if part.lower() in {"train", "val", "valid", "validation", "test"}), None)
            if split and split.lower() in {"train", "val", "valid", "validation"}:
                grouped["train" if split.lower() == "train" else "val"].append(image)
        if grouped["train"] or grouped["val"]:
            return grouped["train"], grouped["val"]

    shuffled = list(images)
    random.Random(42).shuffle(shuffled)
    split_index = max(1, int(len(shuffled) * train_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def prepare_dataset(dataset: Path, output_dir: Path, train_ratio: float = 0.8) -> Path:
    images = find_image_files(dataset)
    if not images:
        raise FileNotFoundError(f"No image files were found in dataset: {dataset}")

    train_images, val_images = create_split(images, dataset, train_ratio=train_ratio)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_images in {"train": train_images, "val": val_images}.items():
        for image_path in split_images:
            relative = image_path.relative_to(dataset)
            destination_image = output_dir / "images" / split_name / relative.name
            destination_label = output_dir / "labels" / split_name / relative.name.replace(image_path.suffix, ".txt")
            copy_file(image_path, destination_image)
            label_path = find_label_path(image_path, dataset)
            if label_path is not None:
                copy_file(label_path, destination_label)

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "path: " + str(output_dir),
                "train: images/train",
                "val: images/val",
                "nc: 6",
                "names: [\"Deformation\", \"Obstacle\", \"Rupture\", \"Disconnect\", \"Misalignment\", \"Deposition\"]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(train_images)} training images and {len(val_images)} validation images.")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/raw"), help="Where the raw dataset lives.")
    parser.add_argument("--output", type=Path, default=Path("data/processed"), help="Where to write the YOLO-formatted dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio between 0 and 1.")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset path does not exist: {args.dataset}")
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1.")

    prepare_dataset(args.dataset, args.output, args.train_ratio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
