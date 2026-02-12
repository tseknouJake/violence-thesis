#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(folder: Path):
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Audit dataset structure and class balance.")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="Path to dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write JSON report.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    violence_dir = dataset_dir / "Violence"
    non_violence_dir = dataset_dir / "NoViolence"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not violence_dir.exists() or not non_violence_dir.exists():
        raise FileNotFoundError(
            "Expected class folders not found. Required: 'Violence' and 'NoViolence'"
        )

    v_files = list_videos(violence_dir)
    nv_files = list_videos(non_violence_dir)
    total = len(v_files) + len(nv_files)

    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "class_counts": {"Violence": len(v_files), "NoViolence": len(nv_files), "total": total},
        "class_ratio": {
            "Violence": round(len(v_files) / total, 4) if total else 0.0,
            "NoViolence": round(len(nv_files) / total, 4) if total else 0.0,
        },
        "sample_files": {
            "Violence": [str(p.relative_to(dataset_dir)) for p in v_files[:5]],
            "NoViolence": [str(p.relative_to(dataset_dir)) for p in nv_files[:5]],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    print("Dataset audit complete.")
    print(json.dumps(report["class_counts"], indent=2))
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
