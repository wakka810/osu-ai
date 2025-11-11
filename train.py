# train.py
from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aim_transformer.config import TrainingConfig
from aim_transformer.data import CoordinateNormalizer, OsuAimDataset
from aim_transformer.model import AimTransformer
from aim_transformer.utils import ModelEMA
from aim_transformer.user_settings import training_config


# -----------------------
# CLI
# -----------------------
def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer-based osu! aim predictor")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoint path for resume/fine-tune (defaults come from user_settings.py).",
    )
    return parser.parse_args()


def _to_serializable(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def config_to_wandb(cfg: TrainingConfig) -> Dict[str, Any]:
    return _to_serializable(asdict(cfg))


def select_maps(config: TrainingConfig) -> tuple[List[str], List[str]]:
    if config.train_maps and config.val_maps:
        return list(config.train_maps), list(config.val_maps)
    maps = sorted([p.name for p in config.data_root.iterdir() if p.is_dir()])
    if len(maps) < 2:
        raise ValueError("Need at least two maps to create train/val splits")
    rng = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(maps), generator=rng).tolist()
    maps = [maps[i] for i in perm]
    return maps[:-1], maps[-1:]


def unwrap_module(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def pick_eval_model(model: nn.Module, ema: Optional[object]) -> nn.Module:
    if ema is None:
        return unwrap_module(model)
    for attr in ("ema", "model", "model_ema"):
        cand = getattr(ema, attr, None)
        if isinstance(cand, nn.Module):
            return unwrap_module(cand)
    return unwrap_module(model)


def build_scheduler(optimizer: torch.optim.Optimizer, max_epochs: int, warmup_epochs: int):
    warmup_epochs = max(0, int(warmup_epochs))
    max_epochs = int(max_epochs)

    def lr_lambda(epoch: int):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_amp_dtype(config: TrainingConfig, device: torch.device):
    if device.type != "cuda":
        return None, False  # CPU: FP32
    amp_choice = config.amp_dtype
    if amp_choice == "bf16":
        return torch.bfloat16, False
    if amp_choice == "fp16":
        return torch.float16, True
    if amp_choice == "fp32":
        return None, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    else:
        return torch.float16, True


# -----------------------
# Main
# -----------------------
def main() -> None:
    cli = parse_cli()
    config = training_config()
    if cli.checkpoint is not None:
        config.checkpoint_path = cli.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.resolve_paths()
    config.validate_runtime(device=device.type)
    config.mixed_precision = device.type == "cuda" and config.amp_dtype != "fp32"

    # normalize types that may arrive as lists from user edits
    config.model.temporal_dilations = tuple(config.model.temporal_dilations)

    wandb_run = None
    if config.use_wandb:
        try:
            import wandb

            wandb_kwargs: Dict[str, Any] = {}
            if config.wandb_project:
                wandb_kwargs["project"] = config.wandb_project
            if config.wandb_run_name:
                wandb_kwargs["name"] = config.wandb_run_name
            if config.wandb_entity:
                wandb_kwargs["entity"] = config.wandb_entity
            tags = list(config.wandb_tags) if config.wandb_tags else None
            if tags:
                wandb_kwargs["tags"] = tags
            if config.wandb_mode:
                wandb_kwargs["mode"] = config.wandb_mode
            wandb_run = wandb.init(**wandb_kwargs, config=config_to_wandb(config))
        except ImportError:
            print("[warn] Weights & Biases logging requested but wandb is not installed. Continuing without it.")
        except Exception as exc:
            print(f"[warn] Failed to initialize Weights & Biases: {exc}")
            wandb_run = None

    # perf toggles
    if config.tf32 and device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
            # conv 側の新API設定を使うなら:
            # torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception:
            pass
    if config.cudnn_benchmark and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
        except Exception:
            pass

    config.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    # 設定保存（JSON）
    try:
        config.save_json()
    except Exception:
        pass

    train_maps, val_maps = select_maps(config)
    normalizer = CoordinateNormalizer()

    holdout_fraction = getattr(config, "val_fraction", 0.0)
    manual_split = (config.train_maps is not None) or (config.val_maps is not None)
    use_holdout_fraction = holdout_fraction > 0.0 and not manual_split

    if use_holdout_fraction:
        # 全譜面から一部を検証に回す
        all_maps = sorted(set(train_maps + val_maps))
        train_dataset = OsuAimDataset(
            data_root=config.data_root,
            map_names=all_maps,
            window_size=config.model.window_size,
            target_offset=config.model.target_offset,
            frame_stride=config.model.frame_stride,
            preload_frames=config.preload_frames,
            normalizer=normalizer,
            split="train",
            holdout_fraction=holdout_fraction,
        )
        val_dataset = OsuAimDataset(
            data_root=config.data_root,
            map_names=all_maps,
            window_size=config.model.window_size,
            target_offset=config.model.target_offset,
            frame_stride=config.model.frame_stride,
            preload_frames=False,
            normalizer=normalizer,
            split="val",
            holdout_fraction=holdout_fraction,
        )
    else:
        train_dataset = OsuAimDataset(
            data_root=config.data_root,
            map_names=train_maps,
            window_size=config.model.window_size,
            target_offset=config.model.target_offset,
            frame_stride=config.model.frame_stride,
            preload_frames=config.preload_frames,
            normalizer=normalizer,
            split="train",
        )
        val_dataset = OsuAimDataset(
            data_root=config.data_root,
            map_names=val_maps,
            window_size=config.model.window_size,
            target_offset=config.model.target_offset,
            frame_stride=config.model.frame_stride,
            preload_frames=False,
            normalizer=normalizer,
            split="val",
        )

    dl_common = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=(config.prefetch_factor if config.num_workers > 0 else None),
        persistent_workers=(config.persistent_workers and config.num_workers > 0),
    )

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **dl_common)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_common)

    steps_per_epoch = len(train_loader)

    model = AimTransformer(
        window_size=config.model.window_size,
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        patch_rows=config.model.patch_rows,
        patch_cols=config.model.patch_cols,
        temporal_kernel_size=config.model.temporal_kernel_size,
        temporal_dilations=config.model.temporal_dilations,
        use_temporal_conv=config.model.use_temporal_conv,
        use_cls_token=config.model.use_cls_token,
    ).to(device)

    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=getattr(config.optimizer, "betas", (0.9, 0.999)),
        eps=getattr(config.optimizer, "eps", 1e-8),
    )

    amp_dtype, need_scaler = resolve_amp_dtype(config, device)
    scaler = amp.GradScaler(enabled=need_scaler)
    criterion = nn.SmoothL1Loss()
    ema = ModelEMA(model, config.ema_decay) if config.ema_decay and config.ema_decay > 0 else None
    scheduler = build_scheduler(optimizer, max_epochs=config.scheduler.max_epochs, warmup_epochs=config.scheduler.warmup_epochs)

    start_epoch = 0
    best_val = float("inf")
    no_improve = 0
    global_step = 0

    # Resume
    if (config.resume or (config.checkpoint_path and config.checkpoint_path.exists())):
        ckpt_path = config.checkpoint_path or (config.output_dir / "latest.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_key = "model_state" if "model_state" in ckpt else "model"
            if state_key in ckpt:
                unwrap_module(model).load_state_dict(ckpt[state_key], strict=False)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scaler_state" in ckpt and scaler.is_enabled() and ckpt["scaler_state"] is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            if "best_val" in ckpt:
                best_val = float(ckpt["best_val"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"])
            print(f"Resumed from {ckpt_path} (epoch={start_epoch}, best_val={best_val:.5f})")

    global_step = start_epoch * steps_per_epoch

    # ======== tqdm bars: outer epoch bar + inner train/val bars ========
    total_epochs = config.scheduler.max_epochs
    epoch_bar = tqdm(
        range(start_epoch, total_epochs),
        desc="Epochs",
        total=total_epochs,
        initial=start_epoch,
        position=0,
        leave=True,
        dynamic_ncols=True,
        ascii=True,
        disable=not config.show_progress,
    )

    for epoch in epoch_bar:
        model.train()
        # --- TRAIN BAR (position=1) ---
        train_bar = tqdm(
            train_loader,
            desc=f"Train [{epoch+1}/{total_epochs}]",
            position=1,
            leave=True,
            dynamic_ncols=True,
            ascii=True,
            mininterval=0.2,
            smoothing=0.1,
            disable=not config.show_progress,
        )

        running = 0.0
        count = 0
        epoch_loss_sum = 0.0
        epoch_sample_count = 0

        for step, (frames, targets, _) in enumerate(train_bar, start=1):
            frames = frames.to(device, non_blocking=config.pin_memory)           # (B, T, 1, H, W)
            targets = targets.to(device, non_blocking=config.pin_memory)         # (B, 2)

            # autocast
            if amp_dtype is not None:
                ctx = amp.autocast(device_type="cuda", dtype=amp_dtype)
            else:
                ctx = amp.autocast(device_type=device.type, enabled=False)

            with ctx:
                preds = model(frames)["coords"]  # (B, 2) in [0,1]
                loss = criterion(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if config.grad_clip_norm and config.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.grad_clip_norm and config.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()

            if ema is not None:
                ema.update(model)

            loss_value = float(loss.detach().cpu().item())
            running += loss_value
            count += 1
            bsz = frames.size(0)
            epoch_loss_sum += loss_value * bsz
            epoch_sample_count += bsz
            global_step += 1

            if step % config.log_interval == 0:
                avg = running / max(1, count)
                lr = optimizer.param_groups[0]["lr"]
                train_bar.set_postfix_str(f"loss={avg:.4f} lr={lr:.2e}")
                if wandb_run is not None:
                    wandb_run.log(
                        {"train/loss": avg, "train/lr": lr, "train/epoch": epoch + 1},
                        step=global_step,
                    )
                running = 0.0
                count = 0

        scheduler.step()

        if epoch_sample_count > 0 and wandb_run is not None:
            epoch_avg_loss = epoch_loss_sum / epoch_sample_count
            wandb_run.log(
                {"train/epoch_loss": epoch_avg_loss, "train/epoch": epoch + 1},
                step=global_step,
            )

        # --- VALIDATION (interval) ---
        do_val = ((epoch + 1) % config.val_interval == 0) or (epoch + 1 == total_epochs)
        if do_val:
            eval_model = pick_eval_model(model, ema).eval()
            total_loss = 0.0
            total = 0

            # VAL BAR (position=2)
            val_bar = tqdm(
                val_loader,
                desc=f"Val   [{epoch+1}/{total_epochs}]",
                position=2,
                leave=True,
                dynamic_ncols=True,
                ascii=True,
                mininterval=0.2,
                smoothing=0.1,
                disable=not config.show_progress,
            )

            with torch.inference_mode():
                for frames, targets, _ in val_bar:
                    frames = frames.to(device, non_blocking=config.pin_memory)
                    targets = targets.to(device, non_blocking=config.pin_memory)
                    preds = eval_model(frames)["coords"]
                    loss = criterion(preds, targets)
                    bsz = frames.size(0)
                    total_loss += float(loss.detach().cpu().item()) * bsz
                    total += bsz

            val_loss = total_loss / max(1, total)
            print(f"Epoch {epoch+1}: val_loss={val_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.6f}")
            if wandb_run is not None:
                wandb_run.log(
                    {"val/loss": val_loss, "train/epoch": epoch + 1},
                    step=global_step,
                )

            # Save latest + best
            ckpt_model = pick_eval_model(model, ema)
            state = unwrap_module(ckpt_model).state_dict()
            payload = {
                "epoch": epoch + 1,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                "best_val": best_val,
            }
            latest_path = config.output_dir / "latest.pt"
            best_path = config.output_dir / "best.pt"
            torch.save(payload, latest_path)

            if val_loss + config.early_stopping_min_delta < best_val:
                best_val = val_loss
                no_improve = 0
                payload["best_val"] = best_val
                torch.save(payload, best_path)
                if wandb_run is not None:
                    wandb_run.summary["best_val"] = best_val
            else:
                no_improve += 1

            if config.early_stopping_patience > 0 and no_improve >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1} (best_val={best_val:.5f})")
                if wandb_run is not None:
                    wandb_run.log({"events/early_stop_epoch": epoch + 1}, step=global_step)
                break

    print("Done.")
    if wandb_run is not None:
        wandb_run.summary["best_val"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
