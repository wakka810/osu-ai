from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class DataSplit:
    train_maps: List[str]
    val_maps: List[str]

    def validate(self) -> None:
        if not self.train_maps:
            raise ValueError("At least one training map is required.")
        if not self.val_maps:
            raise ValueError("At least one validation map is required.")


@dataclass
class ModelConfig:
    window_size: int = 6
    target_offset: int = 1
    frame_stride: int = 1
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    patch_rows: int = 4
    patch_cols: int = 5
    temporal_kernel_size: int = 3
    temporal_dilations: Sequence[int] = (1, 2)
    use_temporal_conv: bool = True
    use_cls_token: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    betas: Sequence[float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 5
    max_epochs: int = 50


@dataclass
class TrainingConfig:
    # I/O
    data_root: Path = Path("datasets/output")
    output_dir: Path = Path("runs/aim_transformer")
    checkpoint_path: Optional[Path] = None
    resume: bool = False

    # Runtime / control
    channels_last: bool = False
    cudnn_benchmark: bool = False
    amp_dtype: str = "auto"
    tf32: bool = False
    use_wandb: bool = False
    wandb_project: str = "aim-transformer"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: Sequence[str] = field(default_factory=tuple)
    wandb_mode: Optional[str] = None
    preload_frames: bool = False
    show_progress: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Loader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2

    # Train misc
    grad_clip_norm: float = 1.0
    log_interval: int = 50
    val_interval: int = 1
    mixed_precision: bool = True
    seed: int = 42
    ema_decay: float = 0.0
    val_fraction: float = 0.1

    # Nested configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Optional explicit splits
    train_maps: Optional[Sequence[str]] = None
    val_maps: Optional[Sequence[str]] = None

    def resolve_paths(self) -> None:
        self.data_root = Path(self.data_root).expanduser()
        self.output_dir = Path(self.output_dir).expanduser()
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path).expanduser()

    def validate_runtime(self, device: Optional[str] = None) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be >= 0")
        allowed_amp = {"auto", "bf16", "fp16", "fp32"}
        if self.amp_dtype not in allowed_amp:
            raise ValueError(f"amp_dtype must be one of {sorted(allowed_amp)}")
        if self.use_wandb:
            if not self.wandb_project:
                raise ValueError("wandb_project must be provided when use_wandb is True")
            allowed_modes = {None, "online", "offline", "disabled", "dryrun"}
            if self.wandb_mode not in allowed_modes:
                raise ValueError(f"wandb_mode must be one of {sorted(m for m in allowed_modes if m is not None)} or None")
        if self.num_workers == 0:
            # 実際の DataLoader では prefetch_factor/persistent は無効相当
            self.persistent_workers = False
        if device is not None and device != "cuda":
            self.mixed_precision = False
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in the range [0, 1).")

    def split_from_dir(self, seed: int = 42) -> None:
        """train_maps/val_maps が未指定なら data_root 下のディレクトリ名から自動分割（最後の1マップを val）。"""
        if self.train_maps is not None and self.val_maps is not None:
            return
        maps = sorted([p.name for p in Path(self.data_root).iterdir() if p.is_dir()])
        if len(maps) < 2:
            raise ValueError("Need at least two maps to create train/val splits")
        rnd = random.Random(seed)
        rnd.shuffle(maps)
        self.train_maps, self.val_maps = maps[:-1], maps[-1:]

    def save_json(self, path: Optional[Path] = None) -> Path:
        out = path or (self.output_dir / "config.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2, default=str)
        return out


@dataclass
class InferenceConfig:
    checkpoint_path: Path = Path("runs/aim_transformer/latest.pt")
    device: str = "auto"
    window_name: str = "osu!"
    frame_width: int = 160
    frame_height: int = 120
    frame_skip: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)

    def resolve_paths(self) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(self.checkpoint_path).expanduser()
