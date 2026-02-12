import torch
import torch.nn as nn
from torchvision.models import video as tv_video


def make_model(name="r3d_18", num_classes=2, pretrained=True):
    name = name.lower()

    if name in {"r3d_18", "mc3_18", "r2plus1d_18", "s3d"}:
        weights = "KINETICS400_V1" if pretrained else None
        model = getattr(tv_video, name)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in {"swin3d_t", "swin3d_s", "swin3d_b"}:
        weights = "KINETICS400_V1" if pretrained else None
        model = getattr(tv_video, name)(weights=weights)
        if hasattr(model, "head") and isinstance(model.head, nn.Linear):
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            raise RuntimeError("Unexpected Video Swin head structure.")
        return model

    if name == "slowfast_r50":
        try:
            from pytorchvideo.models.hub import slowfast_r50
        except Exception as exc:
            raise ImportError(
                "slowfast_r50 requires pytorchvideo. Install with `pip install pytorchvideo`."
            ) from exc
        model = slowfast_r50(pretrained=pretrained)
        if hasattr(model, "blocks") and hasattr(model.blocks[-1], "proj"):
            in_features = model.blocks[-1].proj.in_features
            model.blocks[-1].proj = nn.Linear(in_features, num_classes)
        elif hasattr(model, "head") and hasattr(model.head, "proj"):
            in_features = model.head.proj.in_features
            model.head.proj = nn.Linear(in_features, num_classes)
        else:
            raise RuntimeError("Unexpected SlowFast head structure.")
        return model

    raise ValueError(f"Unknown model: {name}")
