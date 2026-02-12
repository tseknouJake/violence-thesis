#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def collect_labeled_files(dataset_dir: Path):
    rows = []
    classes = [("Violence", 1), ("NoViolence", 0)]
    for class_name, label in classes:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append(
                    {
                        "path": str(p.resolve()),
                        "rel_path": str(p.relative_to(dataset_dir)),
                        "class_name": class_name,
                        "label": label,
                    }
                )
    if not rows:
        raise RuntimeError("No video files found.")
    return rows


def stratified_split(rows, seed: int):
    rng = random.Random(seed)
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    out = []
    for label, group in by_label.items():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        train_part = group[:n_train]
        val_part = group[n_train : n_train + n_val]
        test_part = group[n_train + n_val :]

        for item in train_part:
            item = item.copy()
            item["split"] = "train"
            out.append(item)
        for item in val_part:
            item = item.copy()
            item["split"] = "val"
            out.append(item)
        for item in test_part:
            item = item.copy()
            item["split"] = "test"
            out.append(item)
    return out


def write_csv(path: Path, rows):
    fieldnames = ["path", "rel_path", "class_name", "label", "split"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_counts(rows, key):
    counts = {}
    for row in rows:
        val = row[key]
        counts[val] = counts.get(val, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Create deterministic stratified train/val/test splits.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = collect_labeled_files(args.dataset_dir)
    split_df = stratified_split(df, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        split_rows = [r for r in split_df if r["split"] == split_name]
        write_csv(args.output_dir / f"{split_name}.csv", split_rows)

    summary = {
        "seed": args.seed,
        "total": len(split_df),
        "by_split": value_counts(split_df, "split"),
        "by_class": value_counts(split_df, "class_name"),
        "by_split_and_class": {
            split: value_counts([r for r in split_df if r["split"] == split], "class_name")
            for split in ["train", "val", "test"]
        },
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print("Split generation complete.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
