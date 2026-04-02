import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== YOLOv11核心模块 =====================
def autopad(k, p=None, d=1):
    """自动计算padding以保持输出尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积模块：Conv + BN + SiLU"""

    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """标准瓶颈模块"""

    def __init__(self, in_channels, out_channels, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(out_channels * e)
        self.cv1 = Conv(in_channels, c_, k[0], 1)
        self.cv2 = Conv(c_, out_channels, k[1], 1, g=g)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """YOLOv8/11的C2f模块"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(out_channels * e)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2(C2f):
    """YOLOv11的C3k2模块"""

    def __init__(self, in_channels, out_channels, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(in_channels, out_channels, n, shortcut, g, e)
        if c3k:
            self.m = nn.ModuleList([C3k(self.c, self.c, 2, shortcut, g) for _ in range(n)])


class C3k(nn.Module):
    """C3k瓶颈模块"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(out_channels * e)
        self.cv1 = Conv(in_channels, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, out_channels, 1)
        self.m = nn.ModuleList([Bottleneck(self.c_, self.c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """快速空间金字塔池化"""

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class C2PSA(nn.Module):
    """YOLOv11的C2PSA模块（带空间注意力）"""

    def __init__(self, in_channels, out_channels, n=1, e=0.5):
        super().__init__()
        self.c = int(out_channels * e)
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, out_channels, 1)
        self.m = nn.ModuleList([PSA(self.c, self.c) for _ in range(n)])

    def forward(self, x):
        x = self.cv1(x)
        a, b = x.split((self.c, self.c), 1)
        b = self.m[0](b)
        return self.cv2(torch.cat([a, b], 1))


class PSA(nn.Module):
    """空间注意力模块（修正版）"""

    def __init__(self, in_channels, out_channels, e=0.5):
        super().__init__()
        c_ = int(in_channels * e)
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.attn = nn.MultiheadAttention(c_, 4, batch_first=True)
        self.cv3 = Conv(c_ * 2, out_channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        feat1 = self.cv1(x)
        feat2 = self.cv2(x)

        q = feat1.flatten(2).permute(0, 2, 1)
        k = feat2.flatten(2).permute(0, 2, 1)
        v = k

        attn_out, _ = self.attn(q, k, v)
        attn_out = attn_out.permute(0, 2, 1).reshape(b, -1, h, w)

        combined = torch.cat([feat1, attn_out], dim=1)
        out = self.cv3(combined)
        return out


class DySample(nn.Module):
    """动态采样模块（轻量级上采样）"""

    def __init__(self, in_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, in_channels * scale * scale, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.pixel_shuffle(self.conv(x))


class SegHead(nn.Module):
    """语义分割头：将特征图映射到类别数"""

    def __init__(self, in_channels, num_classes, inter_channels=128):
        super().__init__()
        self.conv1 = Conv(in_channels, inter_channels, 3, 1)
        self.conv2 = Conv(inter_channels, inter_channels, 3, 1)
        self.conv3 = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


# ===================== 基于YOLOv11的语义分割模型 =====================
class YOLOv11Segmenter(nn.Module):
    """
    基于YOLOv11架构的语义分割模型
    """

    def __init__(self, in_channels=4, num_classes=6, dropout_rate=0.1):
        super().__init__()

        # Stem
        self.stem = Conv(in_channels, 64, 3, 2)

        # Stage 1: 64 -> 128
        self.stage1 = nn.Sequential(
            Conv(64, 128, 3, 2),
            C3k2(128, 128, n=2, c3k=False)
        )

        # Stage 2: 128 -> 256
        self.stage2 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3k2(256, 256, n=2, c3k=True)
        )

        # Stage 3: 256 -> 512
        self.stage3 = nn.Sequential(
            Conv(256, 512, 3, 2),
            C3k2(512, 512, n=2, c3k=True)
        )

        # Stage 4: 512 -> 1024
        self.stage4 = nn.Sequential(
            Conv(512, 1024, 3, 2),
            C3k2(1024, 1024, n=2, c3k=True)
        )

        # SPPF模块
        self.sppf = SPPF(1024, 1024, k=5)

        # C2PSA模块
        self.c2psa = C2PSA(1024, 1024, n=1)

        # 解码器
        self.decoder1 = nn.ModuleDict({
            'up': DySample(1024, scale=2),
            'conv': Conv(1024 + 512, 512, 3, 1)
        })

        self.decoder2 = nn.ModuleDict({
            'up': DySample(512, scale=2),
            'conv': Conv(512 + 256, 256, 3, 1)
        })

        self.decoder3 = nn.ModuleDict({
            'up': DySample(256, scale=2),
            'conv': Conv(256 + 128, 128, 3, 1)
        })

        self.decoder4 = nn.ModuleDict({
            'up': DySample(128, scale=2),
            'conv': Conv(128 + 64, 64, 3, 1)
        })

        # 最终上采样
        self.final_up = DySample(64, scale=2)
        self.final_conv = Conv(64, 64, 3, 1)

        # Dropout层
        self.dropout = nn.Dropout2d(dropout_rate)

        # 语义分割头
        self.seg_head = SegHead(64, num_classes, inter_channels=64)

    def forward(self, x):
        # 记录输入尺寸
        input_size = x.shape[-2:]

        # 编码器
        x1 = self.stem(x)  # 1/2, 64
        x2 = self.stage1(x1)  # 1/4, 128
        x3 = self.stage2(x2)  # 1/8, 256
        x4 = self.stage3(x3)  # 1/16, 512
        x5 = self.stage4(x4)  # 1/32, 1024
        x5 = self.sppf(x5)
        x5 = self.c2psa(x5)

        # 解码器
        d1 = self.decoder1['up'](x5)
        d1 = torch.cat([d1, x4], dim=1)
        d1 = self.decoder1['conv'](d1)

        d2 = self.decoder2['up'](d1)
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.decoder2['conv'](d2)

        d3 = self.decoder3['up'](d2)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.decoder3['conv'](d3)

        d4 = self.decoder4['up'](d3)
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.decoder4['conv'](d4)

        # 最终上采样
        out = self.final_up(d4)
        out = self.final_conv(out)
        out = self.dropout(out)

        # 语义分割预测
        out = self.seg_head(out)

        # 关键修复：确保输出尺寸与输入一致
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        return out


# 保持与原模型兼容的类名
RemoteSensingSegmenter = YOLOv11Segmenter

