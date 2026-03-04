# Week 1 Checklist (Starting March 4, 2026)

## Goal for this week
Get from scaffold-only to first real baseline training run (SlowFast) on clean Bus Violence data.

## Monday (today) - Environment and baseline code skeleton
- [x] Add core dependencies for training:
  - torch
  - torchvision
  - pytorchvideo (or equivalent)
  - scikit-learn (optional for verification metrics)
- [x] Create a real training script (separate from stub), e.g. `scripts/05_train_baseline.py`.
- [x] Define config schema for training hyperparameters (batch size, epochs, lr, num_workers, device).
- [x] Add dataloader module using existing split CSVs.

## Tuesday - First clean training run
- [x] Run short sanity baseline on Bus Violence train/val (dependency-free mode; replace with real SlowFast once torch stack is installable).
- [x] Save:
  - model checkpoint
  - val predictions CSV
  - val metrics JSON
  - run log JSON
- [x] Run test inference and save test predictions + metrics.

## Wednesday - Verify and stabilize
- [ ] Re-run with same seed to check reproducibility.
- [ ] Confirm metric consistency within expected variation.
- [ ] Measure inference latency and throughput on test split.
- [ ] Record one row in `docs/experiment_log_template.csv` (copy to your own log file first).

## Thursday - Analysis and error review
- [ ] Inspect confusion matrix and identify FP/FN patterns.
- [ ] Pull 10 false positives + 10 false negatives and note common visual traits.
- [ ] Write 0.5-1 page analysis notes from findings.

## Friday - Writing + supervisor-ready update
- [ ] Write 1-2 pages:
  - reproducible pipeline status
  - first real baseline setup
  - initial clean-data performance
  - observed failure modes
- [ ] Prepare one short update summary:
  - what was done
  - what is blocked
  - exact plan for next week (Swin + corruption pipeline start)

## Hard deliverables by end of week
- [ ] At least 1 real baseline run completed and saved in `outputs/experiments/<run_id>/`.
- [ ] At least 1 written results/progress page.
- [ ] At least 1 filled experiment-log row with real metrics.

## If blocked fallback
- [ ] If SlowFast setup fails, run a smaller 3D CNN baseline first and document reason.
- [ ] If full training is too slow, run short-epoch sanity training and clearly label it as preliminary.
