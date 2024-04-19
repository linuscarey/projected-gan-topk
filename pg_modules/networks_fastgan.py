# original implementation: https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import torch.nn as nn
import torch
from pg_modules.blocks import (InitLayer, UpBlockBig, UpBlockBigCond, UpBlockSmall, UpBlockSmallCond, SEBlock, conv2d)
from pg_modules.topk_loss_module import find_topk_operation_using_name

def normalize_second_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class DummyMapping(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, c, **kwargs):
        return z.unsqueeze(1)  # to fit the StyleGAN API


class FastganSynthesis(nn.Module):
    def __init__(self, ngf=128, z_dim=256, nc=3, img_resolution=256, lite=False, sparse_hw_info=None):
        super().__init__()
        self.img_resolution = img_resolution
        self.z_dim = z_dim
        self.sparse_hw_info = sparse_hw_info

        # channel multiplier
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        # layers
        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmall if lite else UpBlockBig

        self.feat_8   = UpBlock(nfc[4], nfc[8])
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlock(nfc[16], nfc[32])
        self.feat_64  = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlock(nfc[64], nfc[128])
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        # sparse layers
        if self.sparse_hw_info != "None":
            self.sparse_hw_reso, self.sparse_hw_topk = self.sparse_hw_info.split("_")
            self.sparse_hw_reso = [int(i) for i in self.sparse_hw_reso.split("-")]
            self.sparse_hw_topk = [int(i) * 0.01 for i in self.sparse_hw_topk.split("-")]
            self.sparse_hw_topk_info = {self.sparse_hw_reso[i]: self.sparse_hw_topk[i] for i in range(len(self.sparse_hw_reso))}
            for reso in self.sparse_hw_reso:
                topk_keep_num = max(int(self.sparse_hw_topk_info[reso] * reso * reso), 1)
                setattr(self, f"sparse_hw_layer_{reso}", find_topk_operation_using_name(args.sp_hw_policy_name)(self.sparse_hw_topk_info[reso], topk_keep_num, reso * reso, args, reso))
        else:
            self.sparse_hw_topk_info = "None"

        self.sparse_layer_loss = []

        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])
    
    def sparse_layer(self, reso, x):
        tau = 1.

        if self.sparse_hw_topk_info != "None" and reso in self.sparse_hw_topk_info:
            sparse_layer_reso = getattr(self, f"sparse_hw_layer_{reso}")
            x, loss = sparse_layer_reso(x, tau = tau)
            if torch.is_tensor(loss):
                self.sparse_layer_loss.append(loss.unsqueeze(0))
        return x

    def forward(self, input, c, inject_noise=0, **kwargs):
        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_4 = self.sparse_layer(4, feat_4)
        feat_8 = self.feat_8(feat_4)
        feat_8 = self.sparse_layer(8, feat_8)
        feat_16 = self.feat_16(feat_8)
        feat_16 = self.sparse_layer(16, feat_16)
        feat_32 = self.feat_32(feat_16)
        feat_32 = self.sparse_layer(32, feat_32)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32))
        if inject_noise:
            eval_time_noise = torch.rand(feat_64.shape).cuda() * inject_noise
            feat_64 += eval_time_noise
        feat_64 = self.sparse_layer(64, feat_64)

        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64))
        feat_128 = self.sparse_layer(128, feat_128)

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last))
            feat_last = self.sparse_layer(256, feat_last)

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last))
            feat_last = self.sparse_layer(512, feat_last)

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last)
            feat_last = self.sparse_layer(1024, feat_last)

        return self.to_big(feat_last)


class FastganSynthesisCond(nn.Module):
    def __init__(self, ngf=64, z_dim=256, nc=3, img_resolution=256, num_classes=1000, lite=False, sparse_hw_info=None):
        super().__init__()

        self.z_dim = z_dim
        nfc_multi = {2: 16, 4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5,
                     512:0.25, 1024:0.125, 2048:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.img_resolution = img_resolution
        self.sparse_hw_info = sparse_hw_info

        self.init = InitLayer(z_dim, channel=nfc[2], sz=4)

        UpBlock = UpBlockSmallCond if lite else UpBlockBigCond

        self.feat_8   = UpBlock(nfc[4], nfc[8], z_dim)
        self.feat_16  = UpBlock(nfc[8], nfc[16], z_dim)
        self.feat_32  = UpBlock(nfc[16], nfc[32], z_dim)
        self.feat_64  = UpBlock(nfc[32], nfc[64], z_dim)
        self.feat_128 = UpBlock(nfc[64], nfc[128], z_dim)
        self.feat_256 = UpBlock(nfc[128], nfc[256], z_dim)

        # sparse layers
        if self.sparse_hw_info != "None":
            self.sparse_hw_reso, self.sparse_hw_topk = self.sparse_hw_info.split("_")
            self.sparse_hw_reso = [int(i) for i in self.sparse_hw_reso.split("-")]
            self.sparse_hw_topk = [int(i) * 0.01 for i in self.sparse_hw_topk.split("-")]
            self.sparse_hw_topk_info = {self.sparse_hw_reso[i]: self.sparse_hw_topk[i] for i in range(len(self.sparse_hw_reso))}
            for reso in self.sparse_hw_reso:
                topk_keep_num = max(int(self.sparse_hw_topk_info[reso] * reso * reso), 1)
                setattr(self, f"sparse_hw_layer_{reso}", find_topk_operation_using_name(args.sp_hw_policy_name)(self.sparse_hw_topk_info[reso], topk_keep_num, reso * reso, args, reso))
        else:
            self.sparse_hw_topk_info = "None"

        self.sparse_layer_loss = []

        self.se_64 = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_big = conv2d(nfc[img_resolution], nc, 3, 1, 1, bias=True)

        if img_resolution > 256:
            self.feat_512 = UpBlock(nfc[256], nfc[512])
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if img_resolution > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])

        self.embed = nn.Embedding(num_classes, z_dim)

    def sparse_layer(self, reso, x):
        tau = 1.

        if self.sparse_hw_topk_info != "None" and reso in self.sparse_hw_topk_info:
            sparse_layer_reso = getattr(self, f"sparse_hw_layer_{reso}")
            x, loss = sparse_layer_reso(x, tau = tau)
            if torch.is_tensor(loss):
                self.sparse_layer_loss.append(loss.unsqueeze(0))
        return x

    def forward(self, input, c, inject_noise=0, update_emas=False):
        c = self.embed(c.argmax(1))

        # map noise to hypersphere as in "Progressive Growing of GANS"
        input = normalize_second_moment(input[:, 0])

        feat_4 = self.init(input)
        feat_4 = self.sparse_layer(4, feat_4)
        feat_8 = self.feat_8(feat_4, c)
        feat_8 = self.sparse_layer(8, feat_8)
        feat_16 = self.feat_16(feat_8, c)
        feat_16 = self.sparse_layer(16, feat_16)
        feat_32 = self.feat_32(feat_16, c)
        feat_32 = self.sparse_layer(32, feat_32)

        feat_64 = self.se_64(feat_4, self.feat_64(feat_32, c))
        if inject_noise:
            eval_time_noise = torch.rand(feat_64.shape).cuda() * inject_noise
            feat_64 += eval_time_noise
        feat_64 = self.sparse_layer(64, feat_64)

        feat_128 = self.se_128(feat_8,  self.feat_128(feat_64, c))
        feat_128 = self.sparse_layer(128, feat_128)

        if self.img_resolution >= 128:
            feat_last = feat_128

        if self.img_resolution >= 256:
            feat_last = self.se_256(feat_16, self.feat_256(feat_last, c))
            feat_last = self.sparse_layer(256, feat_last)

        if self.img_resolution >= 512:
            feat_last = self.se_512(feat_32, self.feat_512(feat_last, c))
            feat_last = self.sparse_layer(512, feat_last)

        if self.img_resolution >= 1024:
            feat_last = self.feat_1024(feat_last, c)
            feat_last = self.sparse_layer(1024, feat_last)

        return self.to_big(feat_last)


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=256,
        c_dim=0,
        w_dim=0,
        img_resolution=256,
        img_channels=3,
        ngf=128,
        cond=0,
        mapping_kwargs={},
        synthesis_kwargs={}
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        # Mapping and Synthesis Networks
        self.mapping = DummyMapping()  # to fit the StyleGAN API
        Synthesis = FastganSynthesisCond if cond else FastganSynthesis
        self.synthesis = Synthesis(ngf=ngf, z_dim=z_dim, nc=img_channels, img_resolution=img_resolution, **synthesis_kwargs)

    def forward(self, z, c, **kwargs):
        w = self.mapping(z, c)
        img = self.synthesis(w, c)
        return img
