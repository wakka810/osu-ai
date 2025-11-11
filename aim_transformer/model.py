from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn


class FrameEncoder(nn.Module):
    def __init__(self, embed_dim: int, patch_rows: int, patch_cols: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.proj = nn.Conv2d(128, embed_dim, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((patch_rows, patch_cols))
        self.num_patches = patch_rows * patch_cols

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, T, C=1, H, W)
        b, t, c, h, w = frames.shape
        x = frames.reshape(b * t, c, h, w)  # non-contiguous 安全
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B*T, P, D)
        x = x.reshape(b, t, self.num_patches, -1)  # (B, T, P, D)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, dilations: Sequence[int]) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        layers = []
        for dilation in dilations:
            padding = dilation * (kernel_size - 1) // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=embed_dim),
                    nn.GELU(),
                    nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*P, T, D)
        out = x.transpose(1, 2)  # (B*P, D, T)
        for block in self.layers:
            residual = out
            out = block(out) + residual
        out = out.transpose(1, 2)
        out = self.norm(out + x)
        return out


class AimTransformer(nn.Module):
    def __init__(
        self,
        window_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        patch_rows: int,
        patch_cols: int,
        temporal_kernel_size: int = 3,
        temporal_dilations: Sequence[int] = (1, 2),
        use_temporal_conv: bool = True,
        use_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        self.encoder = FrameEncoder(embed_dim=embed_dim, patch_rows=patch_rows, patch_cols=patch_cols)
        self.embed_dim = embed_dim
        self.num_patches = self.encoder.num_patches
        self.use_cls_token = use_cls_token

        if use_temporal_conv:
            self.temporal_conv = TemporalConvBlock(embed_dim, temporal_kernel_size, temporal_dilations)
        else:
            self.temporal_conv = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # nested tensor 警告を明示的にオフ（norm_first=True と相性問題のため）
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=depth, enable_nested_tensor=False
        )

        # Learnable positional embeddings（小さめ初期化）
        with torch.no_grad():
            self.frame_positional = nn.Parameter(torch.empty(window_size, 1, embed_dim).normal_(mean=0.0, std=0.02))
            self.patch_positional = nn.Parameter(torch.empty(1, self.num_patches, embed_dim).normal_(mean=0.0, std=0.02))

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cls_positional = nn.Parameter(torch.empty(1, 1, embed_dim))
            nn.init.normal_(self.cls_positional, mean=0.0, std=0.02)
        else:
            self.cls_token = None
            self.cls_positional = None

        self.coord_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

        # (frames, device_type) -> mask
        self._mask_cache: Dict[Tuple[int, str, bool], torch.Tensor] = {}

    def _build_causal_mask(self, frames: int, device: torch.device) -> torch.Tensor:
        """
        Build and cache a causal attention mask for the given frame length. The mask is cloned on
        retrieval so that individual forward passes do not share mutable buffers.
        """
        key = (frames, device.type, self.use_cls_token)
        if key in self._mask_cache:
            mask = self._mask_cache[key]
            if mask.device == device:
                # キャッシュから出すときは clone ではなく、呼び出し側で clone する運用にしているが
                # 念のためここでも安全性を確保しておく
                return mask.clone()
        # フレーム単位で未来フレームを遮断（同一フレーム内のパッチは相互に可視）
        frame_ids = torch.arange(frames, device=device).repeat_interleave(self.num_patches)
        future = frame_ids.unsqueeze(0) < frame_ids.unsqueeze(1)  # True: future
        min_value = torch.finfo(torch.float32).min
        base_tokens = frames * self.num_patches
        base_mask = torch.zeros((base_tokens, base_tokens), dtype=torch.float32, device=device)
        base_mask.masked_fill_(future, min_value)

        if self.use_cls_token:
            total_tokens = base_tokens + 1
            mask = torch.zeros((total_tokens, total_tokens), dtype=torch.float32, device=device)
            mask[1:, 1:] = base_mask
            mask[1:, 0] = min_value  # prevent leak from future frames via CLS
            # cls row already zeros -> attends everywhere
        else:
            mask = base_mask
        # clone して独立バッファを作る（CUDAGraph 上書き検出回避）
        mask = mask.clone()
        self._mask_cache[key] = mask
        return mask

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        # frames: (B, T, 1, H, W)
        b, t, _, _, _ = frames.shape
        if t > self.window_size:
            raise ValueError(f"Received {t} frames but model window_size={self.window_size}")

        tokens = self.encoder(frames)  # (B, T, P, D)

        if self.temporal_conv is not None and t > 1:
            bsz, t_len, num_patches, dim = tokens.shape
            tokens_conv = tokens.permute(0, 2, 1, 3).contiguous().reshape(bsz * num_patches, t_len, dim)
            tokens_conv = self.temporal_conv(tokens_conv)
            tokens = tokens_conv.reshape(bsz, num_patches, t_len, dim).permute(0, 2, 1, 3).contiguous()

        # Add positional encodings
        frame_pos = self.frame_positional[:t].transpose(0, 1).unsqueeze(2)  # (1, T, 1, D)
        patch_pos = self.patch_positional.unsqueeze(1)  # (1, 1, P, D)
        tokens = (tokens + frame_pos + patch_pos).reshape(b, t * self.num_patches, self.embed_dim)  # (B, TP, D)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            cls_tokens = cls_tokens + self.cls_positional
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        # build mask (cache safe) and clone again as extra保険
        attn_mask = self._build_causal_mask(t, tokens.device).clone()  # (TP, TP)
        encoded = self.transformer(tokens, mask=attn_mask)  # (B, TP, D)

        if self.use_cls_token:
            pooled = encoded[:, 0]  # (B, D)
        else:
            pooled = encoded.mean(dim=1)  # (B, D)
        coord_pred = torch.sigmoid(self.coord_head(pooled))  # (B, 2) in [0,1]

        return {"coords": coord_pred}
