import random
from typing import Iterable, Optional, Union

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def _as_list(value: Union[str, Iterable[str], None]):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def apply_corruption(
    frames: torch.Tensor,
    corruption: Union[str, Iterable[str], None],
    severity: int = 1,
    rng: Optional[random.Random] = None,
):
    """
    Apply one or more corruptions to a video tensor.
    frames: Tensor[T, C, H, W] in [0, 1] float32.
    """
    rng = rng or random
    corruptions = _as_list(corruption)
    if not corruptions:
        return frames

    severity = int(max(1, min(3, severity)))

    for name in corruptions:
        if name == "downsample":
            frames = _downsample(frames, severity)
        elif name == "low_light":
            frames = _low_light(frames, severity)
        elif name == "gaussian_blur":
            frames = _gaussian_blur(frames, severity)
        elif name == "motion_blur":
            frames = _motion_blur(frames, severity)
        elif name == "occlusion":
            frames = _occlusion(frames, severity, rng)
        elif name == "compression":
            frames = _compression(frames, severity)
        elif name == "camera_shake":
            frames = _camera_shake(frames, severity, rng)
        else:
            raise ValueError(f"Unknown corruption: {name}")

    return frames.clamp(0.0, 1.0)


def _downsample(frames: torch.Tensor, severity: int):
    scales = {1: 0.75, 2: 0.5, 3: 0.25}
    scale = scales[severity]
    t, c, h, w = frames.shape
    h2, w2 = max(1, int(h * scale)), max(1, int(w * scale))
    small = F.interpolate(frames, size=(h2, w2), mode="bilinear", align_corners=False)
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def _low_light(frames: torch.Tensor, severity: int):
    factors = {1: 0.7, 2: 0.5, 3: 0.3}
    return frames * factors[severity]


def _gaussian_blur(frames: torch.Tensor, severity: int):
    kernel = {1: 3, 2: 5, 3: 7}[severity]
    sigma = {1: 0.6, 2: 1.2, 3: 1.8}[severity]
    out = []
    for frame in frames:
        out.append(TF.gaussian_blur(frame, kernel_size=[kernel, kernel], sigma=[sigma, sigma]))
    return torch.stack(out, dim=0)


def _motion_blur(frames: torch.Tensor, severity: int):
    k = {1: 3, 2: 5, 3: 9}[severity]
    t, c, h, w = frames.shape
    weight = torch.zeros((c, 1, 1, k), device=frames.device, dtype=frames.dtype)
    weight[:, 0, 0, :] = 1.0 / k
    padding = (0, k // 2)
    return F.conv2d(frames, weight, padding=padding, groups=c)


def _occlusion(frames: torch.Tensor, severity: int, rng: random.Random):
    t, c, h, w = frames.shape
    frac = {1: 0.15, 2: 0.3, 3: 0.45}[severity]
    box_h = max(1, int(h * frac))
    box_w = max(1, int(w * frac))
    out = frames.clone()
    for i in range(t):
        y = rng.randint(0, max(0, h - box_h))
        x = rng.randint(0, max(0, w - box_w))
        out[i, :, y : y + box_h, x : x + box_w] = 0.0
    return out


def _compression(frames: torch.Tensor, severity: int):
    qualities = {1: 50, 2: 25, 3: 10}
    quality = qualities[severity]
    out = []
    for frame in frames:
        img = (frame * 255.0).clamp(0, 255).to(torch.uint8)
        try:
            from torchvision.io import encode_jpeg, decode_jpeg
        except Exception as exc:
            raise ImportError(
                "compression corruption requires torchvision.io encode_jpeg/decode_jpeg."
            ) from exc
        buf = encode_jpeg(img, quality=quality)
        decoded = decode_jpeg(buf).to(torch.float32) / 255.0
        out.append(decoded)
    return torch.stack(out, dim=0)


def _camera_shake(frames: torch.Tensor, severity: int, rng: random.Random):
    max_shift = {1: 2, 2: 5, 3: 10}[severity]
    out = []
    for frame in frames:
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        out.append(
            TF.affine(frame, angle=0.0, translate=[dx, dy], scale=1.0, shear=[0.0, 0.0], fill=0)
        )
    return torch.stack(out, dim=0)
