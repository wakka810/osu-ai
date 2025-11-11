from __future__ import annotations

import argparse
import ctypes
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import cv2
import keyboard
import numpy as np
import torch
import win32gui
from mss import mss

from aim_transformer.config import ModelConfig
from aim_transformer.model import AimTransformer
from aim_transformer.user_settings import inference_config


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime playback for transformer model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override checkpoint path (defaults come from user_settings.py).",
    )
    return parser.parse_args()


def build_model(cfg: ModelConfig) -> AimTransformer:
    cfg.temporal_dilations = tuple(cfg.temporal_dilations)
    model = AimTransformer(
        window_size=cfg.window_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        patch_rows=cfg.patch_rows,
        patch_cols=cfg.patch_cols,
        temporal_kernel_size=cfg.temporal_kernel_size,
        temporal_dilations=cfg.temporal_dilations,
        use_temporal_conv=cfg.use_temporal_conv,
        use_cls_token=cfg.use_cls_token,
    )
    return model


def load_checkpoint(
    model: AimTransformer, checkpoint: Path, device: torch.device, use_half: bool = False
) -> AimTransformer:
    state = torch.load(checkpoint, map_location=device)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    if use_half:
        model = model.half()
    model.eval()
    return model


def find_window(title: str) -> Optional[int]:
    hwnd = win32gui.FindWindow(None, title)
    return hwnd or None


def get_client_region(hwnd: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    screen_left, screen_top = win32gui.ClientToScreen(hwnd, (0, 0))
    return screen_left, screen_top, right - left, bottom - top


def crop_and_resize(frame: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, float, float, float, float]:
    if frame.shape[2] == 4:
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        bgr = frame
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    window_aspect = w / max(h, 1)
    target_aspect = target_w / max(target_h, 1)

    offset_x = 0.0
    offset_y = 0.0
    effective_w = float(w)
    effective_h = float(h)

    if window_aspect > target_aspect:
        new_w = int(round(h * target_aspect))
        offset_x_val = max(0, int(round((w - new_w) / 2)))
        gray = gray[:, offset_x_val : offset_x_val + new_w]
        offset_x = float(offset_x_val)
        effective_w = float(new_w)
    elif window_aspect < target_aspect:
        new_h = int(round(w / target_aspect))
        offset_y_val = max(0, int(round((h - new_h) / 2)))
        gray = gray[offset_y_val : offset_y_val + new_h, :]
        offset_y = float(offset_y_val)
        effective_h = float(new_h)

    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0, effective_w, offset_x, effective_h, offset_y


def move_cursor(x: float, y: float) -> None:
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def main() -> None:
    cli = parse_cli()
    cfg = inference_config()
    if cli.checkpoint is not None:
        cfg.checkpoint_path = cli.checkpoint

    cfg.resolve_paths()
    device_name = cfg.device.lower()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)} (dtype=fp16)")
    else:
        print("Using CPU for inference (dtype=float32)")
    model = build_model(cfg.model)
    model = load_checkpoint(model, Path(checkpoint_path), device, use_half=use_cuda)
    model_dtype = next(model.parameters()).dtype

    frame_shape = (
        1,
        cfg.model.window_size,
        1,
        cfg.frame_height,
        cfg.frame_width,
    )
    frame_buffer = torch.empty(frame_shape, device=device, dtype=model_dtype)

    # Warm-up to let CUDA/cudnn choose optimal kernels before timing
    if use_cuda:
        frame_buffer.zero_()
        warmup_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with warmup_ctx():
            for _ in range(20):
                model(frame_buffer)
        torch.cuda.synchronize(device)

    hwnd = find_window(cfg.window_name)
    if hwnd is None:
        print(f"Window '{cfg.window_name}' not found.")
        return

    screen_left, screen_top, client_w, client_h = get_client_region(hwnd)
    monitor_region = {"left": screen_left, "top": screen_top, "width": client_w, "height": client_h}

    frame_queue: Deque[np.ndarray] = deque(maxlen=cfg.model.window_size)
    frame_skip = max(1, cfg.frame_skip)

    warmup_skip = 50 if use_cuda else 0

    with mss() as sct:
        try:
            counter = 0
            import time

            fps_timer = time.perf_counter()
            fps_counter = 0
            fps_display = 0.0
            capture_total = 0.0
            prep_total = 0.0
            infer_total = 0.0
            frame_count = 0
            active_count = 0

            inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad

            with inference_ctx():
                while True:
                    try:
                        if keyboard.is_pressed("q"):
                            break
                    except RuntimeError:
                        # keyboard library can raise when stdin is detached; ignore and keep running.
                        pass

                    counter += 1
                    if counter % frame_skip != 0:
                        continue

                    t_start = time.perf_counter()
                    frame = np.array(sct.grab(monitor_region), dtype=np.uint8)
                    t_capture = time.perf_counter()
                    processed, eff_w, offset_x, eff_h, offset_y = crop_and_resize(
                        frame, cfg.frame_width, cfg.frame_height
                    )

                    frame_queue.append(processed)
                    if len(frame_queue) < cfg.model.window_size:
                        continue

                    frames_np = np.stack(frame_queue, axis=0)
                    frames_src = torch.from_numpy(frames_np).unsqueeze(1).to(dtype=model_dtype)
                    frame_buffer.copy_(frames_src, non_blocking=use_cuda)
                    t_prep = time.perf_counter()

                    coords = model(frame_buffer)["coords"].squeeze(0).detach().cpu().numpy()
                    t_infer = time.perf_counter()

                    target_x = float(coords[0] * eff_w + offset_x)
                    target_y = float(coords[1] * eff_h + offset_y)

                    abs_x = screen_left + target_x
                    abs_y = screen_top + target_y
                    abs_x = max(screen_left, min(screen_left + client_w - 1, abs_x))
                    abs_y = max(screen_top, min(screen_top + client_h - 1, abs_y))
                    move_cursor(abs_x, abs_y)

                    capture_time = t_capture - t_start
                    prep_time = t_prep - t_capture
                    infer_time = t_infer - t_prep

                    frame_count += 1
                    if frame_count <= warmup_skip:
                        continue

                    capture_total += capture_time
                    prep_total += prep_time
                    infer_total += infer_time
                    active_count += 1

                    fps_counter += 1
                    now = time.perf_counter()
                    elapsed = now - fps_timer
                    if elapsed > 0.0:
                        fps_display = fps_counter / elapsed
                        if elapsed >= 1.0:
                            fps_counter = 0
                            fps_timer = now

                    print(
                        f"FPS: {fps_display:.1f}  capture={capture_time*1e3:.2f}ms  prep={prep_time*1e3:.2f}ms  infer={infer_time*1e3:.2f}ms",
                        end="\r",
                        flush=True,
                    )
        except KeyboardInterrupt:
            pass
    print()
    if active_count > 0:
        avg_capture = capture_total / active_count
        avg_prep = prep_total / active_count
        avg_infer = infer_total / active_count
        print(
            f"Average over {active_count} frames -> capture={avg_capture*1e3:.2f}ms  prep={avg_prep*1e3:.2f}ms  infer={avg_infer*1e3:.2f}ms"
        )


if __name__ == "__main__":
    main()
