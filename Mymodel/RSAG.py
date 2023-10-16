import torch
import torch.nn as nn
from Mymodel import common

def make_model(opt):
    return RSAG(opt)

class RSAG(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(RSAG, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.att = [common.SAttention(n_feats * pow(2, p-1)) for p in  range(self.phase,0, -1)]
        self.att = nn.ModuleList(self.att)
        self.upsample = nn.Upsample(scale_factor=max(opt.scale), mode='bicubic', align_corners=False)
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)
        self.head = conv(opt.n_colors, n_feats, kernel_size)
        self.up = []
        for p in range(self.phase, 0, -1):
            self.up.append(
                conv(opt.n_colors, n_feats * pow(2, p-1), kernel_size)
            )

        self.up = nn.ModuleList(self.up)

        self.down = [
            common.DownBlock(opt, 2, n_feats * pow(2, p), n_feats * pow(2, p), n_feats * pow(2, p + 1)
            ) for p in range(self.phase)
        ]
        self.down = nn.ModuleList(self.down)
        self.down_lf = [common.DownBlock(opt, 2, n_feats, n_feats, n_feats * 2)]
        self.down_lf = nn.ModuleList(self.down_lf)

        up_body_blocks = [[
            common.CBAM(
                conv, 4*n_feats * pow(2, (p-1)), kernel_size, act=act
            ) for _ in range(n_blocks)
        ] for p in range(self.phase, 1, -1)
        ]
        up_body_blocks.insert(0, [
            common.CBAM(
                conv, n_feats * pow(2, self.phase), kernel_size, act=act
            ) for _ in range(n_blocks)
        ])
        up = [[
            common.Upsampler(conv, 2, n_feats * pow(2, self.phase), act=False),
            conv(n_feats * pow(2, self.phase), n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        for p in range(self.phase - 1, 0, -1):    #2,1
            up.append([
                common.Upsampler(conv, 2, 4 * n_feats * pow(2, p), act=False),
                conv(4 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks[idx], *up[idx])
            )

        self.up_lf =  common.DeconvPReLu(n_feats * 4, n_feats, 5, stride=2, padding=2)
        self.tail = conv(4 * n_feats , opt.n_colors, kernel_size)
        self.tail_lf = conv(n_feats * 2, opt.n_colors, kernel_size)
        self.conv35 = common.DCN()

    def forward(self, low ,rgb):
        low_up = self.upsample(low)
        hr_0, lf_0 = self.conv35(low_up)

        x = self.head(hr_0)
        rgb = self.head(rgb)
        lf_res0 = self.head(lf_0)

        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        copie_rgb = []
        for idx in range(self.phase):
            copie_rgb.append(rgb)
            rgb = self.down[idx](rgb)

        copies_depth = []
        for idx in range(self.phase):
            low = self.upsample_2(low)
            x_up = self.up[idx](low)
            copies_depth.append(x_up)

        copies_res = []
        x1 = x
        for idx in range(self.phase):
            x = self.up_blocks[idx](x)
            copies_res.append(x)
            rgb_att = self.att[idx](copies_depth[idx], copie_rgb[self.phase - idx - 1])
            x = torch.cat((x, copies[self.phase - idx - 1], rgb_att, copies_depth[idx]), 1)

        sr = self.tail(x)
        lf_res = self.down_lf[0](lf_res0)
        lf_res = torch.cat((lf_res, copies_res[1]), 1)
        lf_res = self.up_lf(lf_res)
        lf_res = torch.cat((lf_res, copies_res[2]), 1)
        lf_res = self.tail_lf(lf_res)
        sr0 = sr + lf_0 + lf_res

        sr1 = self.head(sr0)
        copies_depth = []
        for idx in range(self.phase):
            copies_depth.append(sr1)
            sr1 = self.down[idx](sr1)

        for idx in range(self.phase):
            x1 = self.up_blocks[idx](x1)
            copies_res.append(x1)
            rgb_att = self.att[idx](copies_depth[self.phase - idx - 1], copie_rgb[self.phase - idx - 1])
            x1 = torch.cat((x1, copies[self.phase - idx - 1], rgb_att, copies_depth[self.phase - idx - 1]), 1)

        sr = self.tail(x1)
        lf_res = self.down_lf[0](lf_res0)
        lf_res = torch.cat((lf_res, copies_res[1]), 1)
        lf_res = self.up_lf(lf_res)
        lf_res = torch.cat((lf_res, copies_res[2]), 1)
        lf_res = self.tail_lf(lf_res)
        sr1 = sr + lf_0 + lf_res

        return sr1, sr0

