#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import classification_metrics, roc_auc_from_scores
from src.utils.io_utils import read_csv, write_json


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV and write metrics JSON.")
    parser.add_argument("--pred_csv", type=Path, required=True, help="CSV with y_true,y_pred and optional y_score")
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON output path")
    args = parser.parse_args()

    rows = read_csv(args.pred_csv)
    y_true = [int(r["y_true"]) for r in rows]
    y_pred = [int(r["y_pred"]) for r in rows]
    y_score = [float(r["y_score"]) for r in rows if "y_score" in r and r["y_score"] != ""]

    metrics = classification_metrics(y_true, y_pred)
    if len(y_score) == len(rows):
        metrics["roc_auc"] = roc_auc_from_scores(y_true, y_score)
    else:
        metrics["roc_auc"] = None

    write_json(args.output, metrics)
    print("Metrics written to:", args.output)
    print(metrics)


if __name__ == "__main__":
    main()
