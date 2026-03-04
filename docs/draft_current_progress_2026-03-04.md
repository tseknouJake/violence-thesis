# Draft: Current Progress (as of March 4, 2026)

## 1. Thesis progress status
The project is currently in the transition phase between planning and core experimentation. The thesis plan is completed, and a reproducible experimental scaffold has been implemented. At this stage, the work has established dataset verification, deterministic data splits, standardized metrics, and experiment logging, but full model-level experiments (SlowFast, Video Swin, corruption benchmarking, and robustness-method evaluation) have not yet been executed.

## 2. Completed technical work
A full pipeline scaffold has been created to ensure reproducibility from the start. First, dataset auditing was implemented to verify expected class-folder structure and class balance. The audit confirms that the Bus Violence dataset currently contains 1,400 videos in total, with balanced classes (700 Violence, 700 NoViolence).

Second, deterministic train/validation/test splits were implemented using a fixed seed with stratification by class. The current split is 70/15/15, resulting in 980 train, 210 validation, and 210 test samples, with class balance preserved in each split.

Third, a shared evaluation module was implemented for classification metrics and ROC-AUC scoring. This standardizes metric computation across all future experiments and prevents inconsistency between model runs.

Fourth, an experiment-output scaffold was implemented, including structured run folders and standardized artifacts such as run metadata, prediction files, and metric reports.

## 3. Sanity-check baseline result
A stub majority-classifier baseline was executed to validate the pipeline mechanics end-to-end. This run is not intended as a scientific baseline for thesis conclusions; instead, it confirms that data ingestion, split usage, prediction export, metric computation, and run logging all function correctly.

As expected for a majority-style stub on a balanced binary dataset, performance is near chance level (accuracy approximately 0.50, ROC-AUC approximately 0.50). This outcome is methodologically useful because it verifies that the scaffold is operational before introducing computationally expensive deep-learning baselines.

## 4. Current gap to thesis objectives
The core thesis objectives still require implementation and empirical validation:
- real baseline training and testing (SlowFast and Video Swin),
- multi-dataset evaluation beyond Bus Violence,
- township-style corruption benchmark construction,
- robustness-method design and ablation,
- comparative clean-vs-corrupted analysis including false-positive behavior and practical viability metrics.

## 5. Immediate next step
The immediate next step is to execute the first real clean-data baseline run (SlowFast) within the existing reproducible framework, using the established splits and metric pipeline. This will mark the start of evidence-generating experiments and enable the first meaningful results section.
