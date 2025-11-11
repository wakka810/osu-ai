# osu! Aim Transformer

TransformerベースのosuのAim予測AIモデル


https://github.com/user-attachments/assets/94520d29-bf83-4668-bb25-024d008b4dcc


## セットアップ

```bash
pip install torch torchvision numpy opencv-python mss pywin32 keyboard tqdm
```

## 学習

```bash
python train.py
```

- `datasets/output/` にリプレイデータを配置
- `aim_transformer/user_settings.py` で設定を調整
- 学習済みモデルは `runs/aim_transformer/` に保存

## 推論

```bash
python test_aim.py
```

- osu!を起動してプレイ中に実行
- モデルがリアルタイムでカーソル位置を予測
- `q`キーで終了

## 設定

`aim_transformer/user_settings.py` で以下を調整:

- `window_size`: 予測に使用するフレーム数
- `batch_size`: バッチサイズ
- `embed_dim`, `depth`, `num_heads`: モデルアーキテクチャ
- `learning_rate`: 学習率
