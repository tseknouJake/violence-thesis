import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from datasets import BusViolenceDataset
from models import make_model


def make_loader(args, split):
    ds = BusViolenceDataset(
        root=args.data_root,
        split_file=args.split_file,
        split=split,
        num_frames=args.num_frames,
        img_size=args.img_size,
        corruption=args.corruption,
        severity=args.severity,
        corruption_prob=1.0,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_scores.extend(probs.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    cm = confusion_matrix(all_labels, all_preds).tolist()
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = float("nan")
    return acc, p, r, f1, auc, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/bus-violence")
    parser.add_argument("--split-file", default="splits/bus_violence_splits.json")
    parser.add_argument("--model", default="r3d_18")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(name=args.model)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)

    loader = make_loader(args, "test")
    acc, p, r, f1, auc, cm = evaluate(model, loader, device)
    print(f"Test acc={acc:.3f} p={p:.3f} r={r:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(f"Confusion matrix: {cm}")


if __name__ == "__main__":
    main()
