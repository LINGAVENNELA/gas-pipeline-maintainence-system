#!/usr/bin/env python3
"""Download the baseline dataset without adding it to the repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET = "simplexitypipeline/pipeline-defect-dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def download_with_kagglehub(output: Path) -> Path | None:
    try:
        import kagglehub
    except ImportError:
        return None

    cache_path = Path(kagglehub.dataset_download(DATASET))
    output.mkdir(parents=True, exist_ok=True)
    for item in cache_path.iterdir():
        destination = output / item.name
        if destination.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    return output


def download_with_cli(output: Path) -> Path:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Install kagglehub or the Kaggle CLI, then configure Kaggle credentials."
        )
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(output), "--unzip"],
        check=True,
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        location = download_with_kagglehub(args.output) or download_with_cli(args.output)
    except Exception as exc:
        print(f"Dataset download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dataset downloaded to {location}")
    print("Run: python training/dataset_audit.py --dataset", location)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
