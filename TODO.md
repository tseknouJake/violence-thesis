# Thesis TODO

## Freeze Scope This Week
- [ ] Freeze the task definition to:
  `Binary CCTV-style physical violence / aggressive incident detection under township-like video degradations in South Africa, using public annotated violence/fight datasets as proxies.`
- [ ] Freeze the core thesis goal:
  measure how badly detectors break under township-like degradations, then test whether a robustness method improves reliability without too many false alarms or too much compute cost.
- [ ] Freeze the core datasets:
  - [ ] `Bus Violence` is mandatory.
  - [ ] Pick only one secondary dataset for the main thesis experiments.
  - [ ] Recommended secondary dataset: `RWF-2000` if you want to build on the reproductions you already started.
  - [ ] Alternative secondary dataset: `SCF` if you want to stay closer to the original plan and keep compute lower.
- [ ] Freeze the baseline families:
  - [ ] one reproducible 3D / temporal baseline
  - [ ] one stronger comparison baseline from a different family if feasible
- [ ] Freeze the reduced-compute protocol for each dataset:
  - [ ] use the same reduced train subset for every repo/model on the same dataset
  - [ ] keep validation/test fixed where possible
  - [ ] record the exact split files, seed, and sample counts
- [ ] Freeze what is out of scope:
  - [ ] broad multi-crime taxonomy
  - [ ] collecting ad hoc YouTube CCTV data
  - [ ] reproducing every repo in the spreadsheet

## Already Done
- [x] Thesis plan and Gantt chart created.
- [x] Dataset audit script created (`scripts/01_audit_dataset.py`).
- [x] Reproducible split script created (`scripts/02_make_splits.py`).
- [x] Baseline stub scaffold created (`scripts/03_run_baseline_stub.py`).
- [x] Shared scoring script created (`scripts/04_score_predictions.py`).
- [x] Dependency-free sanity training script created (`scripts/05_train_baseline.py`).
- [x] Shared metric, dataset, and I/O modules created.
- [x] Bus Violence audited and confirmed balanced.
- [x] Bus Violence deterministic train/val/test splits created.
- [x] Sanity baseline runs saved under `outputs/experiments/`.
- [x] Preliminary repo reproductions started and tracked.

## Immediate Next Step
- [ ] Write a 1-page scope-lock note for your supervisor.
- [ ] In that note, answer only:
  - [ ] exact task definition
  - [ ] core datasets
  - [ ] chosen baselines
  - [ ] reduced-compute protocol
  - [ ] what counts as "done" for the thesis experiments
- [ ] After that note is frozen, run the first real clean baseline under the frozen protocol.

## Core Experimental Work

### Clean Baselines
- [ ] Install or enable the real video-model dependency stack.
- [ ] Replace the current sanity trainer with a true video baseline pipeline.
- [ ] Implement and run baseline A on Bus Violence.
- [ ] Implement and run baseline B on Bus Violence.
- [ ] Log accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- [ ] Measure latency, throughput, and memory.
- [ ] Review false positives and false negatives on clean data.

### Second Dataset
- [ ] Integrate the second dataset loader.
- [ ] Use official splits if available; otherwise create fixed seeded splits.
- [ ] Apply the same reduced-compute rule consistently.
- [ ] Run the selected baselines on the second dataset.

### Township Corruption Benchmark
- [ ] Implement downsampling.
- [ ] Implement compression.
- [ ] Implement low-light / brightness reduction.
- [ ] Implement blur.
- [ ] Implement occlusion.
- [ ] Implement FPS reduction.
- [ ] Define mild / medium / severe settings.
- [ ] Visually validate corruption outputs on sample clips.
- [ ] Run clean vs corrupted evaluations.
- [ ] Build performance-drop tables and FP/FN breakdowns.

### Robustness Method
- [ ] Choose one robustness method.
- [ ] Implement it with config switches.
- [ ] Train/evaluate on clean + degraded data.
- [ ] Run ablations.
- [ ] Compare against baselines on robustness, false alarms, and compute cost.

## Writing
- [ ] Do not write the introduction first.
- [ ] Write methods from the actual pipeline you end up using.
- [ ] Write results from saved run artifacts.
- [ ] Use South Africa literature to motivate the problem and discuss deployment limits.
- [ ] Keep the South Africa linkage in motivation/discussion, not in ad hoc data collection.

## Weekly Discipline
- [ ] Start each week by listing the exact runs/configs to execute.
- [ ] End each week with:
  - [ ] updated metrics table
  - [ ] short notes on what worked / failed
  - [ ] 1-2 pages of thesis writing or structured analysis
