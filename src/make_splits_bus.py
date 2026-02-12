# make_splits_bus.py
import os
import json
import random
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
root = repo_root / "data" / "bus-violence"
classes = {"Violence": 1, "NoViolence": 0}
samples = []

for cls_name, label in classes.items():
    folder = root / cls_name
    if not folder.exists():
        print(f"Warning: {folder} doesn't exist!")
        continue
    
    for fname in os.listdir(folder):
        if fname.lower().endswith((".mp4", ".avi", ".mov")):
            rel_path = str(Path(cls_name) / fname)
            samples.append((rel_path, label))

print(f"Found {len(samples)} videos total")

if len(samples) < 30:
    print("Warning: Very few videos found. Check your dataset path.")
else:
    random.seed(42)
    random.shuffle(samples)
    
    n = len(samples)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:]
    }
    
    print(f"Train: {len(splits['train'])} videos")
    print(f"Val: {len(splits['val'])} videos")
    print(f"Test: {len(splits['test'])} videos")
    
    splits_dir = repo_root / "splits"
    splits_dir.mkdir(exist_ok=True)
    with open(splits_dir / "bus_violence_splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    print("✓ Splits saved to splits/bus_violence_splits.json")
