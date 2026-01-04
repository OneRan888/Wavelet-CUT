#!/usr/bin/env python3
"""
WSI / large-image tiler for CUT.

- 输入：sourceA / sourceB 目录（放 WSI 或超大图）
- 输出：<dataroot>/<phase>A 和 <dataroot>/<phase>B
- 文件名：<slide>_x<X>_y<Y>.png  (X,Y 是 level-0 坐标，便于训练时找邻居块)

需要 WSI（svs/ndpi/mrxs 等）时建议装 OpenSlide：
  pip install openslide-python
并确保系统安装 openslide 动态库。
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide  # type: ignore
    _HAS_OPENSLIDE = True
except Exception:
    openslide = None
    _HAS_OPENSLIDE = False

WSI_EXTS = {'.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu', '.tif', '.tiff'}


def is_wsi(path: Path) -> bool:
    return path.suffix.lower() in WSI_EXTS


def list_images(root: Path):
    exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu'}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def patch_is_blank_rgb(arr: np.ndarray, white_ratio_thresh: float, intensity_thresh: int = 220) -> bool:
    if arr.size == 0:
        return True
    white = np.all(arr >= intensity_thresh, axis=-1)
    return float(white.mean()) >= white_ratio_thresh


def tile_one_slide(
    in_path: Path,
    out_dir: Path,
    tile_size: int,
    overlap_ratio: float,
    level: int,
    skip_blank: bool,
    blank_white_ratio: float,
    out_format: str,
    max_patches: int,
):
    slide_name = in_path.stem
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))

    # OpenSlide 路径
    if is_wsi(in_path) and _HAS_OPENSLIDE:
        slide = openslide.OpenSlide(str(in_path))
        w, h = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        xs = list(range(0, max(1, w - tile_size + 1), stride))
        ys = list(range(0, max(1, h - tile_size + 1), stride))
        count = 0

        for y in ys:
            for x in xs:
                if count >= max_patches:
                    slide.close()
                    return count

                x0 = int(round(x * downsample))
                y0 = int(round(y * downsample))
                img = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert('RGB')
                arr = np.array(img)

                if skip_blank and patch_is_blank_rgb(arr, blank_white_ratio):
                    continue

                out_name = f"{slide_name}_x{x0}_y{y0}.{out_format}"
                img.save(out_dir / out_name)
                count += 1

        slide.close()
        return count

    # PIL fallback（会整张图读入内存）
    img = Image.open(in_path).convert('RGB')
    w, h = img.size
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    count = 0

    for y in ys:
        for x in xs:
            if count >= max_patches:
                return count
            patch = img.crop((x, y, x + tile_size, y + tile_size))
            arr = np.array(patch)
            if skip_blank and patch_is_blank_rgb(arr, blank_white_ratio):
                continue
            out_name = f"{slide_name}_x{x}_y{y}.{out_format}"
            patch.save(out_dir / out_name)
            count += 1

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sourceA', type=str, required=True)
    ap.add_argument('--sourceB', type=str, required=True)
    ap.add_argument('--dataroot', type=str, required=True)
    ap.add_argument('--phase', type=str, default='train')

    ap.add_argument('--tile_size', type=int, default=512)
    ap.add_argument('--overlap_ratio', type=float, default=0.25)
    ap.add_argument('--level', type=int, default=0)

    ap.add_argument('--out_format', type=str, default='png', choices=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
    ap.add_argument('--max_patches_per_slide', type=int, default=10_000_000)

    ap.add_argument('--skip_blank', action='store_true')
    ap.add_argument('--blank_white_ratio', type=float, default=0.8)
    ap.add_argument('--verbose', action='store_true')

    args = ap.parse_args()

    sourceA = Path(args.sourceA)
    sourceB = Path(args.sourceB)
    dataroot = Path(args.dataroot)
    outA = dataroot / f"{args.phase}A"
    outB = dataroot / f"{args.phase}B"
    ensure_dir(outA)
    ensure_dir(outB)

    if any(is_wsi(p) for p in list_images(sourceA)) and not _HAS_OPENSLIDE:
        print("Warning: WSI extensions found but OpenSlide is not installed. Falling back to PIL may be slow/fail.")

    totalA = 0
    for p in list_images(sourceA):
        if args.verbose:
            print(f"[A] {p}")
        totalA += tile_one_slide(
            p, outA, args.tile_size, args.overlap_ratio, args.level,
            args.skip_blank, args.blank_white_ratio, args.out_format, args.max_patches_per_slide
        )

    totalB = 0
    for p in list_images(sourceB):
        if args.verbose:
            print(f"[B] {p}")
        totalB += tile_one_slide(
            p, outB, args.tile_size, args.overlap_ratio, args.level,
            args.skip_blank, args.blank_white_ratio, args.out_format, args.max_patches_per_slide
        )

    print(f"Done. Saved patches: A={totalA}, B={totalB}")


if __name__ == '__main__':
    main()