import numpy as np
import torch
import torch.nn.functional as F

from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util


class CUTModel(BaseModel):
    """CUT and FastCUT
    Contrastive Learning for Unpaired Image-to-Image Translation (ECCV 2020)
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for CUT model"""
        parser.add_argument(
            '--CUT_mode',
            type=str,
            default="CUT",
            choices=['CUT', 'cut', 'FastCUT', 'fastcut']
        )

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument(
            '--nce_idt',
            type=util.str2bool,
            nargs='?',
            const=True,
            default=False,
            help='use NCE loss for identity mapping: NCE(G(Y), Y))'
        )
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument(
            '--nce_includes_all_negatives_from_minibatch',
            type=util.str2bool,
            nargs='?',
            const=True,
            default=False,
            help='If True, include negatives from other samples in minibatch when computing contrastive loss.'
        )
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument(
            '--flip_equivariance',
            type=util.str2bool,
            nargs='?',
            const=True,
            default=False,
            help="Enforce flip-equivariance (FastCUT)."
        )

        # Overlap Consistency / Wavelet Regularization
        parser.add_argument('--lambda_OL', type=float, default=0.0,
                            help='weight for overlap consistency loss (requires dataset to return A2/B2)')
        parser.add_argument('--lambda_W', type=float, default=0.0,
                            help='weight for wavelet-domain structural regularization')
        parser.add_argument('--lambda_low', type=float, default=2.0,
                            help='wavelet low-frequency (LL) constraint weight inside L_wavelet')
        parser.add_argument('--lambda_high', type=float, default=0.1,
                            help='wavelet high-frequency (LH/HL/HH) constraint weight inside L_wavelet')
        parser.add_argument('--overlap_ratio', type=float, default=0.25,
                            help='default overlap ratio used when offset is not provided (heuristic)')
        parser.add_argument('--wavelet_luma_only', type=util.str2bool, nargs='?', const=True, default=False,
                            help='if True, compute wavelet loss on luminance channel only')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # losses to print
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.loss_names += ['OL', 'W', 'W_low', 'W_high']

        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:
            self.model_names = ['G']

        # networks
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
            not opt.no_dropout, opt.init_type, opt.init_gain,
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt
        )
        self.netF = networks.define_F(
            opt.input_nc, opt.netF, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt
        )

        # Haar kernels (LL, LH, HL, HH), shape (4,1,2,2)
        haar = torch.tensor([
            [[1.,  1.],
             [1.,  1.]],

            [[1., -1.],
             [1., -1.]],

            [[1.,  1.],
             [-1., -1.]],

            [[1., -1.],
             [-1.,  1.]],
        ], dtype=torch.float32).unsqueeze(1) * 0.5
        # 修改点1：将 register_buffer 改为直接赋值为普通属性
        self.haar_kernels = haar   # 直接存成普通属性

        # placeholders for optional overlap pair
        self.real_A2 = None
        self.A2_offset = None
        self.fake_B2 = None

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt
            )

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = [PatchNCELoss(opt).to(self.device) for _ in self.nce_layers]

            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # optimizer_F will be created in data_dependent_initialize if needed
            self.optimizer_F = None

    def data_dependent_initialize(self, data):
        """
        netF depends on netG feature shapes, so initialize netF with first forward pass.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        if hasattr(self, "real_A2") and self.real_A2 is not None:
            self.real_A2 = self.real_A2[:bs_per_gpu]
            if hasattr(self, "A2_offset") and self.A2_offset is not None and torch.is_tensor(self.A2_offset):
                self.A2_offset = self.A2_offset[:bs_per_gpu]

        self.forward()

        if self.isTrain:
            self.compute_D_loss().backward()
            self.compute_G_loss().backward()

            # Create optimizer_F only if actually used
            if (self.opt.lambda_NCE > 0.0) and (self.opt.netF == 'mlp_sample'):
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2)
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G (and F if needed)
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()

        use_F = (self.opt.netF == 'mlp_sample') and (self.opt.lambda_NCE > 0.0) and (self.optimizer_F is not None)
        if use_F:
            self.optimizer_F.zero_grad()

        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

        if use_F:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps."""
        AtoB = self.opt.direction == 'AtoB'
        src_key = 'A' if AtoB else 'B'
        tgt_key = 'B' if AtoB else 'A'

        self.real_A = input[src_key].to(self.device)
        self.real_B = input[tgt_key].to(self.device)
        self.image_paths = input[f'{src_key}_paths']

        # Overlap: optional second overlapped patch
        src2_key = src_key + '2'
        self.real_A2 = None
        self.A2_offset = None

        if src2_key in input:
            self.real_A2 = input[src2_key].to(self.device)

            off_key = src_key + '2_offset'
            dx_key = src_key + '2_dx'
            dy_key = src_key + '2_dy'

            if off_key in input:
                off = input[off_key]
                if torch.is_tensor(off):
                    self.A2_offset = off.to(self.device)
                else:
                    self.A2_offset = torch.tensor(off, device=self.device)
            elif (dx_key in input) and (dy_key in input):
                dx = input[dx_key]
                dy = input[dy_key]
                if not torch.is_tensor(dx):
                    dx = torch.tensor(dx)
                if not torch.is_tensor(dy):
                    dy = torch.tensor(dy)
                self.A2_offset = torch.stack([dx, dy], dim=-1).to(self.device)

    def forward(self):
        """Run forward pass; called by both <optimize_parameters> and <test>."""
        bs = self.real_A.size(0)

        parts = [self.real_A]
        has_A2 = (self.real_A2 is not None)
        if has_A2:
            parts.append(self.real_A2)

        if self.opt.nce_idt and self.isTrain:
            parts.append(self.real_B)

        self.real = torch.cat(parts, dim=0)

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)

        idx = 0
        self.fake_B = self.fake[idx:idx + bs]
        idx += bs

        if has_A2:
            self.fake_B2 = self.fake[idx:idx + bs]
            idx += bs
        else:
            self.fake_B2 = None

        if self.opt.nce_idt and self.isTrain:
            self.idt_B = self.fake[idx:idx + bs]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake_list = [self.fake_B.detach()]
        if self.fake_B2 is not None:
            fake_list.append(self.fake_B2.detach())
        fake = torch.cat(fake_list, dim=0)

        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        # Real: match fake batch size robustly
        real = self.real_B
        if fake.size(0) != real.size(0):
            reps = int(np.ceil(fake.size(0) / real.size(0)))
            real = real.repeat(reps, 1, 1, 1)[:fake.size(0)]

        self.pred_real = self.netD(real)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """GAN + NCE + optional overlap + optional wavelet losses for the generator"""
        # 1) GAN loss on all generated patches
        fake_list = [self.fake_B]
        if self.fake_B2 is not None:
            fake_list.append(self.fake_B2)
        fake_all = torch.cat(fake_list, dim=0)

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake_all)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # 2) PatchNCE loss
        if self.opt.lambda_NCE > 0.0:
            nce1 = self.calculate_NCE_loss(self.real_A, self.fake_B)
            if self.fake_B2 is not None and self.real_A2 is not None:
                nce2 = self.calculate_NCE_loss(self.real_A2, self.fake_B2)
                self.loss_NCE = 0.5 * (nce1 + nce2)
            else:
                self.loss_NCE = nce1
        else:
            self.loss_NCE = 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        # 3) Overlap Consistency loss
        self.loss_OL = 0.0
        if (self.opt.lambda_OL > 0.0) and (self.fake_B2 is not None):
            self.loss_OL = self.opt.lambda_OL * self.calculate_overlap_loss(
                self.fake_B, self.fake_B2, self.A2_offset
            )

        # 4) Wavelet-domain structural regularization
        self.loss_W = 0.0
        self.loss_W_low = 0.0
        self.loss_W_high = 0.0
        if self.opt.lambda_W > 0.0:
            w1_low, w1_high = self.calculate_wavelet_loss(self.real_A, self.fake_B)
            if self.fake_B2 is not None and self.real_A2 is not None:
                w2_low, w2_high = self.calculate_wavelet_loss(self.real_A2, self.fake_B2)
                w_low = 0.5 * (w1_low + w2_low)
                w_high = 0.5 * (w1_high + w2_high)
            else:
                w_low, w_high = w1_low, w1_high

            self.loss_W_low = w_low
            self.loss_W_high = w_high
            self.loss_W = self.opt.lambda_W * (w_low + w_high)

        self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_OL + self.loss_W
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)
        if self.opt.flip_equivariance and getattr(self, "flipped_for_equivariance", False):
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # Overlap Consistency
    def calculate_overlap_loss(self, fake1, fake2, offsets=None):
        """
        fake1, fake2: (N,C,H,W)
        offsets: optional (N,2) [dx,dy], patch2 is shifted by (dx,dy) relative to patch1.
        If offsets is None: assume right neighbor using overlap_ratio.
        """
        N, _, H, W = fake1.shape
        device = fake1.device

        if offsets is None:
            stride_x = int(round(W * (1.0 - float(self.opt.overlap_ratio))))
            stride_y = 0
            offsets_t = torch.tensor([[stride_x, stride_y]] * N, device=device, dtype=torch.long)
        else:
            offsets_t = offsets.to(device) if torch.is_tensor(offsets) else torch.tensor(offsets, device=device)
            if offsets_t.dim() == 1:
                offsets_t = offsets_t.view(1, 2).repeat(N, 1)
            offsets_t = offsets_t.long()

        loss_sum = 0.0
        cnt = 0
        for i in range(N):
            dx = int(offsets_t[i, 0].item())
            dy = int(offsets_t[i, 1].item())
            o1, o2 = self._extract_overlap(fake1[i:i + 1], fake2[i:i + 1], dx, dy)
            if o1 is None:
                continue
            loss_sum = loss_sum + F.l1_loss(o1, o2)
            cnt += 1

        if cnt == 0:
            return torch.zeros((), device=device)
        return loss_sum / float(cnt)

    @staticmethod
    def _extract_overlap(p1, p2, dx, dy):
        """
        p1,p2: (1,C,H,W), dx/dy offset of p2 relative to p1.
        """
        _, _, H, W = p1.shape
        oh = H - abs(dy)
        ow = W - abs(dx)
        if oh <= 0 or ow <= 0:
            return None, None

        if dy >= 0:
            y1a, y1b = dy, dy + oh
            y2a, y2b = 0, oh
        else:
            y1a, y1b = 0, oh
            y2a, y2b = -dy, -dy + oh

        if dx >= 0:
            x1a, x1b = dx, dx + ow
            x2a, x2b = 0, ow
        else:
            x1a, x1b = 0, ow
            x2a, x2b = -dx, -dx + ow

        o1 = p1[:, :, y1a:y1b, x1a:x1b]
        o2 = p2[:, :, y2a:y2b, x2a:x2b]
        return o1, o2

    # Haar Wavelet Loss
    def calculate_wavelet_loss(self, src, tgt):
        """
        Returns (low_loss, high_loss) without outer lambda_W
        low_loss already multiplied by lambda_low
        high_loss already multiplied by lambda_high
        """
        if self.opt.wavelet_luma_only:
            src = self._rgb_to_luma(src)
            tgt = self._rgb_to_luma(tgt)

        LL_s, LH_s, HL_s, HH_s = self.haar_wavelet_transform(src)
        LL_t, LH_t, HL_t, HH_t = self.haar_wavelet_transform(tgt)

        low = F.l1_loss(LL_s, LL_t) * float(self.opt.lambda_low)
        high = (
            F.l1_loss(LH_s, LH_t) +
            F.l1_loss(HL_s, HL_t) +
            F.l1_loss(HH_s, HH_t)
        ) * float(self.opt.lambda_high)

        return low, high

    def haar_wavelet_transform(self, x):
        """
        x: (N,C,H,W) -> LL/LH/HL/HH each (N,C,H/2,W/2)
        """
        N, C, H, W = x.shape

        if (H % 2) == 1:
            x = x[:, :, :-1, :]
            H -= 1
        if (W % 2) == 1:
            x = x[:, :, :, :-1]
            W -= 1

        # 修改点2：补充 device=x.device，确保张量设备匹配
        k = self.haar_kernels.to(device=x.device, dtype=x.dtype)  # device follows buffer
        weight = k.repeat(C, 1, 1, 1)  # (4C,1,2,2)

        y = F.conv2d(x, weight, stride=2, padding=0, groups=C)  # (N,4C,H/2,W/2)
        y = y.view(N, C, 4, H // 2, W // 2)

        LL = y[:, :, 0, :, :]
        LH = y[:, :, 1, :, :]
        HL = y[:, :, 2, :, :]
        HH = y[:, :, 3, :, :]
        return LL, LH, HL, HH

    @staticmethod
    def _rgb_to_luma(x):
        if x.size(1) == 1:
            return x
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        return 0.299 * r + 0.587 * g + 0.114 * b