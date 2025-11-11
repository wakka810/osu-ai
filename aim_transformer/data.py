from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class CoordinateNormalizer:
    x_scale: float = 1440.0
    y_scale: float = 1080.0

    def normalize(self, x: float, y: float) -> Tuple[float, float]:
        xn = x / self.x_scale
        yn = y / self.y_scale
        xn = float(min(1.0, max(0.0, xn)))
        yn = float(min(1.0, max(0.0, yn)))
        return xn, yn

    def denormalize(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x_norm * self.x_scale, y_norm * self.y_scale


@dataclass
class FrameRecord:
    frame_idx: int
    click: int
    x: int
    y: int
    path: Path
    tensor: Optional[torch.Tensor] = None  # (H, W)


@dataclass
class SongData:
    name: str
    frames: List[FrameRecord]


def _load_npy_float32(path: Path, mmap: bool = True) -> np.ndarray:
    arr = np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def load_song_records(song_dir: Path, preload: bool = False) -> SongData:
    frame_files = sorted(song_dir.glob("*.npy"), key=lambda p: int(p.stem.split(",")[0]))
    frames: List[FrameRecord] = []
    for file_path in frame_files:
        parts = file_path.stem.split(",")
        if len(parts) < 5:
            raise ValueError(f"Unexpected file name format: {file_path.name}")
        frame_idx, click_flag, _, x_coord, y_coord = parts[:5]

        tensor = None
        if preload:
            npy = _load_npy_float32(file_path, mmap=False)  # 実メモリ展開
            tensor = torch.from_numpy(npy).clone().share_memory_()  # (H, W)

        frames.append(
            FrameRecord(
                frame_idx=int(frame_idx),
                click=int(click_flag),
                x=int(x_coord),
                y=int(y_coord),
                path=file_path,
                tensor=tensor,
            )
        )
    if not frames:
        raise ValueError(f"No frame data found in {song_dir}")
    return SongData(name=song_dir.name, frames=frames)


class OsuAimDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        map_names: Sequence[str],
        window_size: int,
        target_offset: int,
        frame_stride: int,
        preload_frames: bool,
        normalizer: Optional[CoordinateNormalizer] = None,
        split: str = "train",
        holdout_fraction: float = 0.0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if target_offset < 0:
            raise ValueError("target_offset must be non-negative.")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive.")
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'.")
        if not (0.0 <= holdout_fraction < 1.0):
            raise ValueError("holdout_fraction must be in the range [0, 1).")

        self.data_root = Path(data_root)
        self.window_size = window_size
        self.target_offset = target_offset
        self.frame_stride = frame_stride  # 窓の開始位置のステップ
        self.preload_frames = preload_frames
        self.normalizer = normalizer or CoordinateNormalizer()
        self.split = split
        self.holdout_fraction = holdout_fraction

        self.songs: List[SongData] = []
        min_required = window_size + target_offset
        for map_name in map_names:
            song_dir = self.data_root / map_name
            if not song_dir.exists():
                raise FileNotFoundError(f"Song directory not found: {song_dir}")
            song_data = load_song_records(song_dir, preload=preload_frames)
            if len(song_data.frames) < min_required:
                continue
            self.songs.append(song_data)

        if not self.songs:
            raise ValueError("No valid songs found for dataset.")

        self.sequence_index: List[Tuple[int, int]] = []
        for song_idx, song in enumerate(self.songs):
            max_start = len(song.frames) - min_required + 1
            starts = list(range(0, max_start, self.frame_stride))
            selected = self._select_starts(starts)
            for start in selected:
                self.sequence_index.append((song_idx, start))
        if not self.sequence_index:
            raise ValueError("No sliding windows generated for dataset; check window/offset/stride settings.")

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, index: int):
        song_idx, start = self.sequence_index[index]
        frames = self.songs[song_idx].frames

        window_records = frames[start : start + self.window_size]
        target_record = frames[start + self.window_size - 1 + self.target_offset]

        frame_tensors = []
        for record in window_records:
            if self.preload_frames and record.tensor is not None:
                ten = record.tensor  # (H, W)
            else:
                npy = _load_npy_float32(record.path, mmap=True)
                # mmap の配列は read-only のことがあり、そのまま from_numpy すると警告が出る。
                # 書き込み不可の場合だけコピーして writable な配列にする（機能的には同じ）。
                if not npy.flags.writeable:
                    npy = np.array(npy, copy=True)
                ten = torch.from_numpy(npy)  # (H, W) ゼロコピー可能なら維持
            frame_tensors.append(ten.unsqueeze(0))  # (1, H, W)

        # (T, 1, H, W) → DataLoader で (B, T, 1, H, W)
        frames_tensor = torch.stack(frame_tensors, dim=0)

        tx, ty = self.normalizer.normalize(target_record.x, target_record.y)
        target = torch.tensor([tx, ty], dtype=torch.float32)

        metadata = {
            "song": self.songs[song_idx].name,
            "frame_idx": target_record.frame_idx,
            "click": target_record.click,
        }
        return frames_tensor, target, metadata

    @property
    def coordinate_normalizer(self) -> CoordinateNormalizer:
        return self.normalizer

    def _select_starts(self, starts: List[int]) -> List[int]:
        if not starts:
            return []
        if self.holdout_fraction <= 0.0:
            return starts

        total = len(starts)
        val_count = max(1, int(round(total * self.holdout_fraction)))
        val_count = min(val_count, total)

        if self.split == "val":
            return starts[-val_count:]

        # split == "train"
        if val_count >= total:
            # 可能なら1件だけ訓練に残す
            if total > 1:
                val_count = total - 1
            else:
                return []
        return starts[: total - val_count]
