from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from src.utils.io_utils import read_csv


@dataclass(frozen=True)
class VideoSample:
    path: str
    rel_path: str
    class_name: str
    label: int
    split: str


class CsvVideoDataset:
    """Simple split loader around the existing CSV schema.

    This keeps training scripts consistent even before full torch Dataset/DataLoader
    integration is added.
    """

    def __init__(self, csv_path: Path | str):
        rows = read_csv(csv_path)
        self.samples: List[VideoSample] = [
            VideoSample(
                path=r["path"],
                rel_path=r["rel_path"],
                class_name=r["class_name"],
                label=int(r["label"]),
                split=r.get("split", ""),
            )
            for r in rows
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[VideoSample]:
        return iter(self.samples)

    def labels(self) -> List[int]:
        return [s.label for s in self.samples]


def batch_iter(samples: List[VideoSample], batch_size: int) -> Iterable[List[VideoSample]]:
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]


def split_counts(samples: List[VideoSample]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in samples:
        out[s.class_name] = out.get(s.class_name, 0) + 1
    return out
