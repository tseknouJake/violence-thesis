import os, json, random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from corruptions import apply_corruption

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False
    decord = None

import cv2

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

        corruption_list = self.corruption if isinstance(self.corruption, (list, tuple)) else [self.corruption]
        fps_drop = "fps_drop" in corruption_list

        if _HAS_DECORD:
            vr = decord.VideoReader(video_path)
            total = len(vr)
            if fps_drop:
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
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                total = 1

            if fps_drop:
                stride = {1: 2, 2: 3, 3: 4}.get(int(self.severity), 2)
                base = list(range(0, total, stride))
                if len(base) == 0:
                    base = list(range(total))
            else:
                base = list(range(total))

            if len(base) >= self.num_frames:
                indices = torch.linspace(0, len(base) - 1, self.num_frames).long().tolist()
                indices = [base[i] for i in indices]
            else:
                indices = base + [base[-1]] * (self.num_frames - len(base))

            frames_list = []
            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    frame = frames_list[-1] if frames_list else torch.zeros(3, 1, 1)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).permute(2, 0, 1)  # C,H,W
                frames_list.append(frame)
            cap.release()
            frames = torch.stack(frames_list, dim=0)  # T,C,H,W
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
