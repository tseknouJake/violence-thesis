# Violence Thesis Starter

This repository is set up to execute your thesis workflow end-to-end in a reproducible way.

## 1) Immediate Next Steps

1. Create/activate your environment.
2. Install dependencies.
3. Run dataset audit.
4. Create fixed train/val/test splits.
5. Start a baseline run (SlowFast first).

## 2) Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3) Verify Dataset

Expected layout for Bus Violence:

```text
data/
  bus-violence/
    Violence/
    NoViolence/
```

Run:

```bash
python scripts/01_audit_dataset.py \
  --dataset_dir data/bus-violence \
  --output outputs/reports/bus_violence_audit.json
```

## 4) Create Reproducible Splits (70/15/15)

```bash
python scripts/02_make_splits.py \
  --dataset_dir data/bus-violence \
  --output_dir outputs/splits \
  --seed 42
```

This writes:

- `outputs/splits/train.csv`
- `outputs/splits/val.csv`
- `outputs/splits/test.csv`
- `outputs/splits/split_summary.json`

## 5) Thesis Execution Roadmap

1. Baseline A: SlowFast on clean videos.
2. Baseline B: Video Swin on clean videos.
3. Build corruption pipeline (resolution/compression/low-light/blur/occlusion/FPS).
4. Evaluate both baselines under mild/medium/severe corruption.
5. Add robustness method (corruption-aware training and/or quality-aware module).
6. Compare clean vs corrupted performance, FP/FN behavior, and latency.
7. Write results chapter while experiments run.

## 8) Run The Experiment Scaffold

Run a standardized baseline stub (sanity check for the full pipeline):

```bash
python scripts/03_run_baseline_stub.py --config configs/baseline_stub_clean.json
```

This writes one run folder in `outputs/experiments/<run_id>/` with:

- `run_log.json` (time, git commit, config snapshot, environment)
- `predictions_val.csv`
- `predictions_test.csv`
- `metrics_val.json`
- `metrics_test.json`

Evaluate any future model predictions with the same metric code:

```bash
python scripts/04_score_predictions.py \
  --pred_csv outputs/experiments/<run_id>/predictions_test.csv \
  --output outputs/experiments/<run_id>/metrics_test_rescored.json
```

Expected columns in prediction CSV:
- `y_true` (0/1)
- `y_pred` (0/1)
- optional `y_score` (probability/confidence for ROC-AUC)

## 9) What To Record Per Run

For every run (clean and corrupted), record:

1. Setup metadata:
- model name + version
- dataset + split files + seed
- preprocessing/transforms
- corruption type + severity (if any)
- hardware (GPU/CPU), batch size, training time

2. Performance metrics:
- accuracy, precision, recall, F1
- ROC-AUC
- confusion matrix (TP, FP, TN, FN)
- false positive rate and false negative rate

3. Practical viability metrics:
- inference latency (ms/video)
- throughput (videos/sec)
- memory usage (if available)

4. Robustness comparisons:
- clean vs mild/medium/severe deltas
- per-corruption performance drop table
- FP increase under each corruption

5. Notes:
- failure mode examples (short clips or IDs)
- any instability, training failures, or anomalies

## 6) Suggested Chapter/Work Packages

1. Literature + evidence mapping.
2. Reproducible pipeline and baselines.
3. Township Corruption Benchmark definition.
4. Robustness method and ablations.
5. Practical viability analysis (accuracy + false alarms + compute cost).

## 7) Weekly Cadence

1. Monday: define run plan + configs.
2. Midweek: execute experiments.
3. Friday: update tables/figures + writing.
4. Supervisor sync with one-page progress note.
