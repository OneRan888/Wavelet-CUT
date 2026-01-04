#!/usr/bin/env python3
"""
Sliding-window inference with weighted fusion for CUT/FastCUT.

- 读取训练好的 checkpoint（复用 TestOptions / create_model 的加载机制）
- 对输入 WSI/大图进行滑窗切块
- 对每个 tile 跑 netG
- 用 Hann 权重窗融合，减少 stitching artifacts

示例：
python tools/wsi_infer_fusion.py \
  --name my_exp --checkpoints_dir ./checkpoints --epoch latest \
  --input /path/to/wsi_or_big.png \
  --output /path/to/out.png \
  --tile_size 512 --overlap_ratio 0.25 --level 0 \
  --gpu_ids 0
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

try:
    import openslide  # type: ignore
    _HAS_OPENSLIDE = True
except Exception:
    openslide = None
    _HAS_OPENSLIDE = False

from options.test_options import TestOptions
from models import create_model

WSI_EXTS = {'.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu', '.tif', '.tiff'}


def is_wsi(path: Path) -> bool:
    return path.suffix.lower() in WSI_EXTS


def get_wsi_dimensions(path: Path, level: int):
    slide = openslide.OpenSlide(str(path))
    w, h = slide.level_dimensions[level]
    down = slide.level_downsamples[level]
    slide.close()
    return (w, h, down)


def hann2d(h: int, w: int) -> np.ndarray:
    wy = np.ones((1,), dtype=np.float32) if h == 1 else np.hanning(h).astype(np.float32)
    wx = np.ones((1,), dtype=np.float32) if w == 1 else np.hanning(w).astype(np.float32)
    win = np.outer(wy, wx)
    win = np.clip(win, 1e-6, None)
    return win.astype(np.float32)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(arr)
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().squeeze(0).clamp(-1, 1)
    t = (t * 0.5 + 0.5) * 255.0
    arr = t.numpy().astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return Image.fromarray(arr)


def main():
    # 先让 TestOptions 解析（它会吃掉仓库已有参数：--name/--epoch/--gpu_ids/...）
    opt = TestOptions().parse()

    # 再解析我们这个脚本自己的参数
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--output', type=str, required=True)
    ap.add_argument('--tile_size', type=int, default=512)
    ap.add_argument('--overlap_ratio', type=float, default=0.25)
    ap.add_argument('--level', type=int, default=0)
    args2, _ = ap.parse_known_args()

    in_path = Path(args2.input)
    out_path = Path(args2.output)
    tile_size = int(args2.tile_size)
    overlap_ratio = float(args2.overlap_ratio)
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    device = model.device
    netG = model.netG

    win = hann2d(tile_size, tile_size)

    if is_wsi(in_path) and _HAS_OPENSLIDE:
        w, h, down = get_wsi_dimensions(in_path, args2.level)
        acc = np.zeros((h, w, 3), dtype=np.float32)
        wacc = np.zeros((h, w, 1), dtype=np.float32)

        xs = list(range(0, max(1, w - tile_size + 1), stride))
        ys = list(range(0, max(1, h - tile_size + 1), stride))

        for y in ys:
            for x in xs:
                x0 = int(round(x * down))
                y0 = int(round(y * down))

                slide = openslide.OpenSlide(str(in_path))
                tile = slide.read_region((x0, y0), args2.level, (tile_size, tile_size)).convert('RGB')
                slide.close()

                tin = pil_to_tensor(tile).to(device)
                with torch.no_grad():
                    tout = netG(tin)
                out_tile = tensor_to_pil(tout)
                out_arr = np.array(out_tile).astype(np.float32)

                acc[y:y + tile_size, x:x + tile_size, :] += out_arr * win[:, :, None]
                wacc[y:y + tile_size, x:x + tile_size, :] += win[:, :, None]

        fused = acc / np.clip(wacc, 1e-6, None)
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        Image.fromarray(fused).save(out_path)
        print(f"Saved: {out_path}")
        return

    # 普通大图（PIL 整张加载）
    img = Image.open(in_path).convert('RGB')
    w, h = img.size
    acc = np.zeros((h, w, 3), dtype=np.float32)
    wacc = np.zeros((h, w, 1), dtype=np.float32)

    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))

    for y in ys:
        for x in xs:
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            if tile.size != (tile_size, tile_size):
                pad = Image.new('RGB', (tile_size, tile_size), (0, 0, 0))
                pad.paste(tile, (0, 0))
                tile = pad

            tin = pil_to_tensor(tile).to(device)
            with torch.no_grad():
                tout = netG(tin)
            out_tile = tensor_to_pil(tout)
            out_arr = np.array(out_tile).astype(np.float32)

            h2 = min(tile_size, h - y)
            w2 = min(tile_size, w - x)

            acc[y:y + h2, x:x + w2, :] += out_arr[:h2, :w2, :] * win[:h2, :w2, None]
            wacc[y:y + h2, x:x + w2, :] += win[:h2, :w2, None]

    fused = acc / np.clip(wacc, 1e-6, None)
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    Image.fromarray(fused).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
