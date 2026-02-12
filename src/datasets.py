import os, json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import decord

from corruptions import apply_corruption

decord.bridge.set_bridge("torch")

class BusViolenceDataset(Dataset):
    def __init__(
        self,
        root,
        split_file,
        split,
        num_frames=32,
        img_size=224,
        corruption=None,
        severity=0,
        corruption_prob=1.0,
    ):
        self.root = Path(root)
        with open(split_file, "r") as f:
            all_splits = json.load(f)
        self.samples = all_splits[split]  # list of (rel_path, label)

        self.num_frames = num_frames
        self.resize = T.Resize((img_size, img_size))
        self.to_float = T.ConvertImageDtype(torch.float32)
        self.normalize = T.Normalize(
            mean=[0.45, 0.45, 0.45],
            std=[0.225, 0.225, 0.225],
        )
        self.corruption = corruption
        self.severity = severity
        self.corruption_prob = corruption_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        video_path = str(self.root / rel_path)

        vr = decord.VideoReader(video_path)
        total = len(vr)
        # simple uniform frame sampling, with optional FPS drop corruption
        corruption_list = self.corruption if isinstance(self.corruption, (list, tuple)) else [self.corruption]
        if "fps_drop" in corruption_list:
            stride = {1: 2, 2: 3, 3: 4}.get(int(self.severity), 2)
            base = torch.arange(0, total, stride)
            if len(base) == 0:
                base = torch.arange(total)
        else:
            base = torch.arange(total)

        if len(base) >= self.num_frames:
            indices = torch.linspace(0, len(base) - 1, self.num_frames).long()
            indices = base[indices]
        else:
            indices = base
            pad = self.num_frames - len(base)
            indices = torch.cat([indices, indices[-1].repeat(pad)])

        frames = vr.get_batch(indices).permute(0, 3, 1, 2)  # T,H,W,C -> T,C,H,W
        frames = self.resize(frames)
        frames = self.to_float(frames)
        if self.corruption and random.random() <= self.corruption_prob:
            corr_list = self.corruption if isinstance(self.corruption, (list, tuple)) else [self.corruption]
            corr_list = [c for c in corr_list if c and c != "fps_drop"]
            if corr_list:
                frames = apply_corruption(frames, corr_list, self.severity)
        frames = self.normalize(frames)

        # output shape: C,T,H,W as expected by many 3D CNNs
        video = frames.permute(1, 0, 2, 3)
        label = torch.tensor(label, dtype=torch.long)
        return video, label
