from __future__ import annotations

from aim_transformer.config import InferenceConfig, TrainingConfig


def training_config() -> TrainingConfig:
    """Return the base TrainingConfig. Edit this function to tweak defaults without CLI flags."""
    cfg = TrainingConfig()
    # Example tweaks (uncomment and edit):
    # cfg.data_root = "datasets/output"
    # cfg.batch_size = 48
    cfg.use_wandb = True
    # cfg.wandb_project = "aim-transformer"
    # cfg.wandb_run_name = "my-run"
    # cfg.wandb_tags = ("debug",)
    return cfg


def inference_config() -> InferenceConfig:
    """Return the realtime inference configuration."""
    cfg = InferenceConfig()
    # Example tweaks:
    # cfg.window_name = "osu!"
    # cfg.model.window_size = 8
    return cfg
