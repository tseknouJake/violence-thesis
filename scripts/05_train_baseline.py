#!/usr/bin/env python3
import argparse
import hashlib
import math
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CsvVideoDataset
from src.eval.metrics import classification_metrics, roc_auc_from_scores
from src.utils.io_utils import write_csv, write_json, read_json


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        return out
    except Exception:
        return "unknown"


def make_run_id(config: dict) -> str:
    stable = (
        f"{config['experiment_name']}_{config['model_name']}_"
        f"{config.get('seed', 0)}_{datetime.now(timezone.utc).isoformat()}"
    )
    digest = hashlib.md5(stable.encode("utf-8")).hexdigest()[:8]
    return f"{config['experiment_name']}_{config['model_name']}_{digest}"


def sigmoid(x: float) -> float:
    x = max(min(x, 20.0), -20.0)
    return 1.0 / (1.0 + math.exp(-x))


def rel_path_features(rel_path: str, dim: int) -> List[float]:
    vec = [0.0] * dim
    # Use only filename stem and strip obvious class tokens to avoid leakage.
    name = Path(rel_path).name.lower()
    name = name.replace("nonviolence", "").replace("violence", "")

    # Hash 3-gram text features from sanitized name to keep this dependency-free.
    for i in range(max(1, len(name) - 2)):
        gram = name[i : i + 3]
        h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0

    # L2 normalize for stable optimization.
    norm_sq = sum(v * v for v in vec)
    if norm_sq > 0:
        norm = math.sqrt(norm_sq)
        vec = [v / norm for v in vec]
    return vec


class LogisticPathModel:
    def __init__(self, dim: int):
        self.dim = dim
        self.weights = [0.0] * dim
        self.bias = 0.0

    def score(self, x: Sequence[float]) -> float:
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def predict_score(self, x: Sequence[float]) -> float:
        return sigmoid(self.score(x))


def fit_logistic_sgd(
    model: LogisticPathModel,
    x_train: List[List[float]],
    y_train: List[int],
    lr: float,
    weight_decay: float,
    epochs: int,
    seed: int,
):
    rng = random.Random(seed)
    n = len(x_train)
    history = []

    for epoch in range(1, epochs + 1):
        idxs = list(range(n))
        rng.shuffle(idxs)
        epoch_loss = 0.0

        for i in idxs:
            x = x_train[i]
            y = float(y_train[i])
            p = model.predict_score(x)

            # Binary cross-entropy + L2 regularization.
            eps = 1e-8
            epoch_loss += -(y * math.log(p + eps) + (1.0 - y) * math.log(1.0 - p + eps))

            grad = p - y
            for j in range(model.dim):
                model.weights[j] -= lr * (grad * x[j] + weight_decay * model.weights[j])
            model.bias -= lr * grad

        history.append({"epoch": epoch, "avg_loss": round(epoch_loss / max(1, n), 6)})

    return history


def make_prediction_rows(dataset: CsvVideoDataset, model: LogisticPathModel, feature_dim: int):
    out = []
    for s in dataset:
        x = rel_path_features(s.rel_path, feature_dim)
        y_score = model.predict_score(x)
        y_pred = 1 if y_score >= 0.5 else 0
        out.append(
            {
                "path": s.path,
                "rel_path": s.rel_path,
                "class_name": s.class_name,
                "y_true": s.label,
                "y_pred": y_pred,
                "y_score": round(float(y_score), 6),
            }
        )
    return out


def evaluate(pred_rows: List[dict]):
    y_true = [int(r["y_true"]) for r in pred_rows]
    y_pred = [int(r["y_pred"]) for r in pred_rows]
    y_score = [float(r["y_score"]) for r in pred_rows]

    metrics = classification_metrics(y_true, y_pred)
    metrics["roc_auc"] = roc_auc_from_scores(y_true, y_score)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate a dependency-free baseline model using split CSV files."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = read_json(args.config)
    run_id = make_run_id(cfg)
    run_dir = ROOT / cfg["output_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    train_ds = CsvVideoDataset(ROOT / cfg["splits"]["train"])
    val_ds = CsvVideoDataset(ROOT / cfg["splits"]["val"])
    test_ds = CsvVideoDataset(ROOT / cfg["splits"]["test"])

    train_cfg = cfg.get("training", {})
    feature_dim = int(train_cfg.get("feature_dim", 256))
    lr = float(train_cfg.get("learning_rate", 0.05))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 8))

    x_train = [rel_path_features(s.rel_path, feature_dim) for s in train_ds]
    y_train = [int(s.label) for s in train_ds]

    model = LogisticPathModel(feature_dim)
    history = fit_logistic_sgd(
        model=model,
        x_train=x_train,
        y_train=y_train,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        seed=seed,
    )

    val_pred = make_prediction_rows(val_ds, model, feature_dim)
    test_pred = make_prediction_rows(test_ds, model, feature_dim)

    val_metrics = evaluate(val_pred)
    test_metrics = evaluate(test_pred)

    write_csv(
        run_dir / "predictions_val.csv",
        val_pred,
        ["path", "rel_path", "class_name", "y_true", "y_pred", "y_score"],
    )
    write_csv(
        run_dir / "predictions_test.csv",
        test_pred,
        ["path", "rel_path", "class_name", "y_true", "y_pred", "y_score"],
    )
    write_json(run_dir / "metrics_val.json", val_metrics)
    write_json(run_dir / "metrics_test.json", test_metrics)

    model_state = {
        "model_type": cfg.get("model_name", "path_logistic"),
        "feature_type": "hashed_char_trigram",
        "feature_dim": feature_dim,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "weights": model.weights,
        "bias": model.bias,
        "train_history": history,
    }
    write_json(run_dir / "model_state.json", model_state)

    finished_at = datetime.now(timezone.utc).isoformat()
    run_log = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": round(time.time() - t0, 3),
        "git_commit": get_git_commit(),
        "config_snapshot": cfg,
        "environment": {
            "python_version": sys.version,
            "notes": "Dependency-free training mode. Replace with torch SlowFast config when deps are installed.",
        },
    }
    write_json(run_dir / "run_log.json", run_log)

    print(f"Run complete: {run_id}")
    print(f"Saved outputs to: {run_dir}")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
