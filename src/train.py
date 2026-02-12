import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import BusViolenceDataset
from models import make_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def make_loader(args, split, batch_size, num_workers=4):
    ds = BusViolenceDataset(
        root=args.data_root,
        split_file=args.split_file,
        split=split,
        num_frames=args.num_frames,
        img_size=args.img_size,
        corruption=args.corruption if split != "train" else args.train_corruption,
        severity=args.severity,
        corruption_prob=args.corruption_prob,
    )
    return DataLoader(ds, batch_size=batch_size,
                      shuffle=(split=="train"),
                      num_workers=num_workers,
                      pin_memory=True)

def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            preds = logits.argmax(1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    return acc, p, r, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/bus-violence")
    parser.add_argument("--split-file", default="splits/bus_violence_splits.json")
    parser.add_argument("--model", default="r3d_18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--corruption", default=None)
    parser.add_argument("--train-corruption", default=None)
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--corruption-prob", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="checkpoints/bus_best.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = make_loader(args, "train", batch_size=args.batch_size)
    val_loader = make_loader(args, "val", batch_size=args.batch_size)

    model = make_model(name=args.model, pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_f1 = 0.0

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for videos, labels in train_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            logits = model(videos)
            loss = criterion(logits, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        acc, p, r, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: val acc={acc:.3f} p={p:.3f} r={r:.3f} f1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.checkpoint)

if __name__ == "__main__":
    main()
