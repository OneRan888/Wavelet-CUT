import os
import random
import re
from PIL import Image

from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset


_COORD_RE = re.compile(
    r'^(?P<slide>.+?)_x(?P<x>\d+)_y(?P<y>\d+)(?:_[^.]*)?\.(?P<ext>png|jpg|jpeg|tif|tiff|bmp)$',
    re.IGNORECASE
)


def _parse_xy_from_filename(path):
    """
    Expect filenames like:
      slideName_x12345_y67890.png
    Returns (slide_id:str, x:int, y:int) or (None, None, None) if not parseable.
    """
    base = os.path.basename(path)
    m = _COORD_RE.match(base)
    if not m:
        return None, None, None
    return m.group('slide'), int(m.group('x')), int(m.group('y'))


class UnalignedDataset(BaseDataset):
    """
    Unaligned/unpaired dataset.

    Added support for returning an overlapped neighbor patch (A2/B2) for overlap consistency loss.
    This dataset ASSUMES you've already pre-tiled WSIs / large images into patches saved in:
      {dataroot}/{phase}A and {dataroot}/{phase}B
    with coordinate-encoded filenames:
      <slide>_x<X>_y<Y>.png

    If a right-neighbor patch exists at (x + stride, y) for the same slide, we return it as A2
    (or B2 if direction=BtoA).
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument(
            '--use_overlap_pair',
            action='store_true',
            help='If set, dataset will also return a right-neighbor overlapped patch (A2/B2) when available.'
        )
        # patch_size used by tiler script; keep here as fallback (avoid duplicate argparse)
        if not any(a.dest == 'patch_size' for a in parser._actions):
            parser.add_argument('--patch_size', type=int, default=512, help='tile/patch size used for WSI tiling (preprocess).')
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self._index_A = self._build_coord_index(self.A_paths) if opt.use_overlap_pair else {}
        self._index_B = self._build_coord_index(self.B_paths) if opt.use_overlap_pair else {}

    @staticmethod
    def _build_coord_index(paths):
        idx = {}
        for p in paths:
            slide, x, y = _parse_xy_from_filename(p)
            if slide is None:
                continue
            idx[(slide, x, y)] = p
        return idx

    @staticmethod
    def _find_right_neighbor_path(index, path, stride_px):
        slide, x, y = _parse_xy_from_filename(path)
        if slide is None:
            return None
        return index.get((slide, x + stride_px, y), None)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc
        AtoB = self.opt.direction == 'AtoB'

        if AtoB:
            input_nc_eff = input_nc
            output_nc_eff = output_nc
        else:
            input_nc_eff = output_nc
            output_nc_eff = input_nc

        # A 与 A2 必须同一套随机参数，保证 overlap offset 在增强后仍一致
        params_A = get_params(self.opt, A_img.size)
        transform_A = get_transform(self.opt, params_A, grayscale=(input_nc_eff == 1))
        A = transform_A(A_img)

        params_B = get_params(self.opt, B_img.size)
        transform_B = get_transform(self.opt, params_B, grayscale=(output_nc_eff == 1))
        B = transform_B(B_img)

        data = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        if self.opt.use_overlap_pair:
            # 右邻居：stride_px 以“切块脚本的 tile_size + overlap_ratio”为准（坐标来自文件名）
            stride_px = int(round(float(self.opt.patch_size) * (1.0 - float(getattr(self.opt, 'overlap_ratio', 0.25)))))
            src_path = A_path if AtoB else B_path
            src_index = self._index_A if AtoB else self._index_B

            neigh_path = self._find_right_neighbor_path(src_index, src_path, stride_px)

            if neigh_path is not None:
                neigh_img = Image.open(neigh_path).convert('RGB')
                neigh_tensor = transform_A(neigh_img) if AtoB else transform_B(neigh_img)

                # dx/dy 在“最终 tensor 空间”里计算（更稳，不怕 resize/crop）
                W = int(neigh_tensor.shape[-1])
                dx = int(round(W * (1.0 - float(getattr(self.opt, 'overlap_ratio', 0.25)))))
                dy = 0

                # 如果发生水平翻转，则右邻居在翻转后会变成左邻居，dx 取反
                if params_A.get('flip', False):
                    dx = -dx

                if AtoB:
                    data['A2'] = neigh_tensor
                    data['A2_offset'] = [dx, dy]
                    data['A2_paths'] = neigh_path
                else:
                    data['B2'] = neigh_tensor
                    data['B2_offset'] = [dx, dy]
                    data['B2_paths'] = neigh_path

        return data

    def __len__(self):
        return max(self.A_size, self.B_size)
