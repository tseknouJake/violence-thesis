#!/usr/bin/env python3
import argparse
import hashlib
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import classification_metrics, roc_auc_from_scores
from src.utils.io_utils import read_csv, read_json, write_csv, write_json


def get_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        return out
    except Exception:
        return "unknown"


def make_run_id(config):
    stable = (
        f"{config['experiment_name']}_{config['model_name']}_"
        f"{config['seed']}_{datetime.now(timezone.utc).isoformat()}"
    )
    digest = hashlib.md5(stable.encode("utf-8")).hexdigest()[:8]
    return f"{config['experiment_name']}_{config['model_name']}_{digest}"


def train_majority_classifier(train_rows):
    positives = sum(int(r["label"]) for r in train_rows)
    negatives = len(train_rows) - positives
    majority = 1 if positives >= negatives else 0
    pos_rate = positives / len(train_rows) if train_rows else 0.5
    return {"majority_label": majority, "pos_rate": pos_rate}


def predict_rows(rows, model_state, strategy):
    out = []
    for r in rows:
        y_true = int(r["label"])
        if strategy == "majority":
            y_pred = model_state["majority_label"]
            y_score = model_state["pos_rate"]
        else:
            raise ValueError(f"Unsupported baseline strategy: {strategy}")

        out.append(
            {
                "path": r["path"],
                "rel_path": r["rel_path"],
                "class_name": r["class_name"],
                "y_true": y_true,
                "y_pred": int(y_pred),
                "y_score": float(y_score),
            }
        )
    return out


def evaluate_predictions(pred_rows):
    y_true = [int(r["y_true"]) for r in pred_rows]
    y_pred = [int(r["y_pred"]) for r in pred_rows]
    y_score = [float(r["y_score"]) for r in pred_rows]
    metrics = classification_metrics(y_true, y_pred)
    auc = roc_auc_from_scores(y_true, y_score)
    metrics["roc_auc"] = auc
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run baseline stub with standardized experiment outputs.")
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON experiment config.")
    args = parser.parse_args()

    cfg = read_json(args.config)
    run_id = make_run_id(cfg)
    run_dir = ROOT / cfg["output_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    train_rows = read_csv(ROOT / cfg["splits"]["train"])
    val_rows = read_csv(ROOT / cfg["splits"]["val"])
    test_rows = read_csv(ROOT / cfg["splits"]["test"])

    model_state = train_majority_classifier(train_rows)
    val_pred = predict_rows(val_rows, model_state, cfg["baseline_strategy"])
    test_pred = predict_rows(test_rows, model_state, cfg["baseline_strategy"])

    val_metrics = evaluate_predictions(val_pred)
    test_metrics = evaluate_predictions(test_pred)

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
    write_json(run_dir / "model_state.json", model_state)

    finished_at = datetime.now(timezone.utc).isoformat()
    run_log = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": round(time.time() - t0, 3),
        "git_commit": get_git_commit(),
        "config_snapshot": cfg,
        "environment": {"python_version": sys.version},
    }
    write_json(run_dir / "run_log.json", run_log)

    print(f"Run complete: {run_id}")
    print(f"Saved outputs to: {run_dir}")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
