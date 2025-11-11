from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from copy import deepcopy


# =========================
# Metric helpers
# =========================

@dataclass
class MetricAverager:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class MetricLogger:
    def __init__(self) -> None:
        self.metrics: Dict[str, MetricAverager] = {}

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.metrics:
            self.metrics[name] = MetricAverager()
        self.metrics[name].update(value, n)

    def averages(self) -> Dict[str, float]:
        return {name: meter.average for name, meter in self.metrics.items()}

    def reset(self) -> None:
        for meter in self.metrics.values():
            meter.reset()


# =========================
# Exponential Moving Average (EMA)
# =========================

class ModelEMA:
    """
    真の EMA 実装:
      ema_param = decay * ema_param + (1 - decay) * model_param

    - decay: 0.0 のときは EMA 無効（update は何もしない）
    - module: EMA を保持する凍結済みモデル（学習時は .update() でのみ更新）
    - ema エイリアス: pick_eval_model() 互換のため self.ema を self.module に張る

    使い方:
      ema = ModelEMA(model, decay=0.999)
      for step ...:
          ...  # optimizer.step() 前後どちらでも可（通常は step 後）
          ema.update(model)
      eval するとき:
          eval_model = ema.ema  # or pick_eval_model から取得
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        device: Optional[torch.device] = None,
        use_fp32: bool = False,
    ) -> None:
        self.decay = float(max(0.0, min(1.0, decay)))
        # 学習とは独立したコピー（勾配不要）
        self.module: nn.Module = deepcopy(model).eval()
        if use_fp32:
            self.module.float()
        for p in self.module.parameters():
            p.requires_grad_(False)
        if device is not None:
            self.module.to(device=device)
        # 互換性のためのエイリアス（pick_eval_model が 'ema' を探す）
        self.ema: nn.Module = self.module
        self.num_updates: int = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """モデルの現在値で EMA を 1 ステップ更新。"""
        if self.decay <= 0.0:
            return
        self.num_updates += 1

        ema_params = list(self.module.parameters())
        mdl_params = list(model.parameters())
        d = self.decay

        # パラメータを EMA で更新
        for e, m in zip(ema_params, mdl_params):
            if not e.dtype.is_floating_point:
                # 量子化など特殊 dtype はコピーにフォールバック
                e.copy_(m)
                continue
            e.mul_(d).add_(m.detach().to(dtype=e.dtype), alpha=(1.0 - d))

        # バッファ（BN の running stats 等）は逐次同期（EMA ではなくコピー）
        for e_buf, m_buf in zip(self.module.buffers(), model.buffers()):
            if e_buf.shape == m_buf.shape:
                e_buf.copy_(m_buf.detach())

    def to(self, *args, **kwargs) -> "ModelEMA":
        """EMA モジュールを明示的にデバイス/ dtype 移動したい場合に使用。"""
        self.module.to(*args, **kwargs)
        return self

    # -------- Checkpoint I/O --------
    def state_dict(self) -> Dict[str, object]:
        """EMA の完全な状態を返す（推奨形式）。"""
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "module": self.module.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        """
        Checkpoint から復元。
        - 新形式: {"decay": float, "num_updates": int, "module": <state_dict>}
        - 旧形式（後方互換）: <state_dict>（= module の state_dict だけ）
        """
        if "module" in state_dict:
            self.decay = float(state_dict.get("decay", self.decay))
            self.num_updates = int(state_dict.get("num_updates", 0))
            self.module.load_state_dict(state_dict["module"])
        else:
            # 後方互換: 純粋な state_dict と見なす
            self.module.load_state_dict(state_dict)
