# Thesis Execution TODO (Jake Lockitch)

## Status Snapshot
- Plan/formal thesis proposal: submitted
- Reproducible scaffold: complete
- Real model training/evaluation: not started
- Corruption benchmark: not started
- Robustness method (novel contribution): not started
- Final analysis + writing: not started

## Phase 0 - Lock Scope (This Week)
- [ ] Freeze thesis scope in 1 page: datasets, baselines, corruption taxonomy, novelty hypothesis.
- [ ] Confirm mandatory deliverables with supervisor: minimum experiments and minimum writing artifacts.
- [ ] Define hard compute budget (GPU hours/week) and storage budget.
- [ ] Define done criteria for each research question (what evidence is enough to answer it).

## Phase 1 - Real Baselines on Clean Data
- [ ] Add framework dependencies (torch, torchvision, pytorchvideo or equivalent).
- [ ] Implement model training/eval pipeline behind config files (no hardcoding).
- [ ] Implement baseline A: SlowFast.
- [ ] Implement baseline B: Video Swin Transformer.
- [ ] Add deterministic seeds and run-level reproducibility metadata.
- [ ] Run clean training/evaluation for each baseline on Bus Violence.
- [ ] Log metrics per run: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- [ ] Add inference viability metrics: latency/video, throughput, GPU memory.

Acceptance criteria:
- [ ] Both baselines produce repeatable metrics from a fresh run.
- [ ] All outputs are stored under outputs/experiments/<run_id>/ with config + metrics + predictions.

## Phase 2 - Multi-Dataset Coverage
- [ ] Add Real-Life Violence Situations (RLVS) dataset integration.
- [ ] Add Surveillance Camera Fight dataset integration.
- [ ] Standardize dataset adapters to same schema (path, label, split).
- [ ] Create reproducible splits for any dataset lacking official splits.
- [ ] Run clean baseline experiments across all selected datasets.

Acceptance criteria:
- [ ] Same training/eval command works across datasets via config only.

## Phase 3 - Township Corruption Benchmark
- [ ] Implement corruption pipeline for video-level transforms:
- [ ] Resolution downsampling
- [ ] Compression artifacts
- [ ] Low-light / brightness reduction
- [ ] Motion blur
- [ ] Occlusion
- [ ] FPS reduction
- [ ] Define severity levels: mild, medium, severe.
- [ ] Validate corruption outputs visually on sample clips.
- [ ] Evaluate both baselines on each corruption x severity condition.
- [ ] Produce performance-drop tables (clean -> corrupted deltas).
- [ ] Produce FP/FN failure mode breakdown by corruption type.

Acceptance criteria:
- [ ] You can reproduce the full corruption benchmark from one command per model.

## Phase 4 - Novel Robustness Method
- [ ] Choose primary robustness method (e.g., corruption-aware training and/or quality-aware module).
- [ ] Implement method with ablation toggles in config.
- [ ] Train/evaluate on clean + corrupted settings.
- [ ] Run ablations to isolate what actually helps.
- [ ] Compare against both baselines on robustness and viability trade-offs.

Acceptance criteria:
- [ ] Method improves robustness on targeted corruptions without unacceptable clean-performance or latency cost.

## Phase 5 - Thesis Evidence Package
- [ ] Build master results table for all datasets, models, and corruption severities.
- [ ] Build key figures: clean-vs-corrupted curves, FP/FN changes, latency trade-offs.
- [ ] Write methods chapter from actual implementation details.
- [ ] Write results chapter from finalized experiment logs.
- [ ] Write discussion: limits, risks, external validity, deployment implications.
- [ ] Cross-check all claims against run artifacts and scripts.

Acceptance criteria:
- [ ] Every table/figure in writing can be traced to a specific run_id.

## Weekly Operating Checklist
- [ ] Monday: decide exact runs/configs for week.
- [ ] Tuesday-Wednesday: execute experiments.
- [ ] Thursday: analyze errors and update tables/plots.
- [ ] Friday: write 1-2 pages from this week's results.
- [ ] Friday: supervisor update with concrete evidence and next-week plan.

## Immediate Next 10 Tasks (Priority Order)
- [ ] 1. Add real training dependency stack to requirements.
- [ ] 2. Build baseline training script interface (train/eval/predict).
- [ ] 3. Implement SlowFast baseline config + run.
- [ ] 4. Implement Video Swin baseline config + run.
- [ ] 5. Add latency/throughput measurement utility.
- [ ] 6. Integrate RLVS dataset loader.
- [ ] 7. Integrate Surveillance Camera Fight dataset loader.
- [ ] 8. Implement corruption transforms + severity config.
- [ ] 9. Run full baseline x corruption matrix.
- [ ] 10. Draft robustness-method design and first ablation plan.
