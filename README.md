Violence Thesis Experiments

This repo contains a baseline pipeline for CCTV-style violence detection and a corruption benchmark.

Quick start (Bus Violence baseline)

1. Generate splits if needed:
   `python src/make_splits_bus.py`
2. Train:
   `python src/train.py --model r3d_18 --data-root data/bus-violence --split-file splits/bus_violence_splits.json`
3. Evaluate:
   `python src/eval.py --model r3d_18 --checkpoint checkpoints/bus_best.pt`
4. Corruption benchmark:
   `python src/benchmark_corruptions.py --model r3d_18 --checkpoint checkpoints/bus_best.pt`

Available models (see `src/models.py`):
- `r3d_18`, `r2plus1d_18`, `mc3_18`, `s3d`
- `swin3d_t`, `swin3d_s`, `swin3d_b`
- `slowfast_r50` (requires `pytorchvideo`)

Corruptions (see `src/corruptions.py`):
- `downsample`, `compression`, `low_light`, `gaussian_blur`, `motion_blur`, `occlusion`, `camera_shake`, `fps_drop`
