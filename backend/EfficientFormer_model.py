"""
遥感4通道语义分割 - 修复版 (解决1024 vs 512尺寸不匹配)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, cast
import math


# ==========================================
# 激活函数定义
# ==========================================
def h_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return F.relu6(x + 3.0, inplace=True) / 6.0

def h_swish(x: torch.Tensor) -> torch.Tensor:
    return x * F.relu6(x + 3.0, inplace=True) / 6.0

class HSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return h_swish(x)

class HSigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return h_sigmoid(x)

class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: float = 0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * reduction_ratio))
        self.conv_reduce = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)
        self.act2 = HSigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.act2(x_se)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ==========================================
# 核心模型：EfficientFormerV2 + UPerNet
# ==========================================
class EfficientFormerV2Block(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        self.token_mixer = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.token_mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class EfficientFormerV2Encoder(nn.Module):
    def __init__(self, in_channels: int = 4, embed_dims: List[int] = [32, 64, 128, 256], depths: List[int] = [2, 2, 6, 4]):
        super().__init__()
        self.patch_embed1 = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
            nn.Conv2d(embed_dims[0], embed_dims[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.GELU(),
        )
        self.patch_embed2 = nn.Sequential(
            nn.Conv2d(embed_dims[0], embed_dims[1], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[1]),
            nn.GELU(),
        )
        self.patch_embed3 = nn.Sequential(
            nn.Conv2d(embed_dims[1], embed_dims[2], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[2]),
            nn.GELU(),
        )
        self.patch_embed4 = nn.Sequential(
            nn.Conv2d(embed_dims[2], embed_dims[3], 3, 2, 1, bias=False),
            nn.BatchNorm2d(embed_dims[3]),
            nn.GELU(),
        )
        self.blocks1 = nn.ModuleList([EfficientFormerV2Block(embed_dims[0]) for _ in range(depths[0])])
        self.blocks2 = nn.ModuleList([EfficientFormerV2Block(embed_dims[1]) for _ in range(depths[1])])
        self.blocks3 = nn.ModuleList([EfficientFormerV2Block(embed_dims[2]) for _ in range(depths[2])])
        self.blocks4 = nn.ModuleList([EfficientFormerV2Block(embed_dims[3]) for _ in range(depths[3])])
        self.out_norms = nn.ModuleList([
            LayerNorm2d(embed_dims[0]),
            LayerNorm2d(embed_dims[1]),
            LayerNorm2d(embed_dims[2]),
            LayerNorm2d(embed_dims[3]),
        ])
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = []
        x = self.patch_embed1(x)
        for blk in self.blocks1: x = blk(x)
        outs.append(self.out_norms[0](x))
        x = self.patch_embed2(x)
        for blk in self.blocks2: x = blk(x)
        outs.append(self.out_norms[1](x))
        x = self.patch_embed3(x)
        for blk in self.blocks3: x = blk(x)
        outs.append(self.out_norms[2](x))
        x = self.patch_embed4(x)
        for blk in self.blocks4: x = blk(x)
        outs.append(self.out_norms[3](x))
        return outs

class LightUPerNetHead(nn.Module):
    def __init__(self, in_channels: List[int] = [32, 64, 128, 256], num_classes: int = 6, channels: int = 128):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, channels, 1, bias=False) for in_ch in in_channels])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(channels, channels, 3, 1, 1, bias=False) for _ in range(len(in_channels))])
        self.cls_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, num_classes, 1)
        )
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        laterals = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, inputs)]
        # 自顶向下融合
        for i in range(len(laterals)-2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(laterals[i+1], scale_factor=2, mode='bilinear', align_corners=False)
        fpn_outs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        # 聚合
        out = fpn_outs[0]
        for i in range(1, len(fpn_outs)):
            out = out + F.interpolate(fpn_outs[i], scale_factor=2**i, mode='bilinear', align_corners=False)
        
        # 分类
        out = self.cls_conv(out)
        # 【关键修复】：这里从 scale_factor=4 改为 scale_factor=2
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return out

class EfficientFormerV2_Seg(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 6):
        super().__init__()
        self.encoder = EfficientFormerV2Encoder(in_channels=in_channels)
        self.decoder = LightUPerNetHead(num_classes=num_classes)
        
        self.register_buffer("pixel_mean", torch.tensor([0.35, 0.38, 0.36, 0.45], dtype=torch.float32).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.18, 0.19, 0.19, 0.22], dtype=torch.float32).view(-1, 1, 1))
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ EfficientFormerV2_Seg 初始化完成，总参数量: {total_params/1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.pixel_mean).to(x.device).to(x.dtype)
        std = cast(torch.Tensor, self.pixel_std).to(x.device).to(x.dtype)
        x = (x - mean) / std
        
        feats = self.encoder(x)
        logits = self.decoder(feats)
        return logits


# ==========================================
# 损失函数
# ==========================================
class CombinedSegLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, weight_ce: float = 1.0, weight_dice: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        
        # Dice Loss
        probs = F.softmax(logits, dim=1)
        loss_dice = 0.0
        valid_mask = targets != self.ignore_index
        for c in range(self.num_classes):
            target_c = (targets == c).float() * valid_mask.float()
            prob_c = probs[:, c, :, :]
            intersection = (prob_c * target_c).sum()
            union = prob_c.sum() + target_c.sum()
            loss_dice += 1 - (2. * intersection + 1e-6) / (union + 1e-6)
        loss_dice /= self.num_classes
        
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice


# ==========================================
# 工厂函数
# ==========================================
def create_remote_seg_model(
    model_type: str = "efficientformer",
    in_channels: int = 4,
    num_classes: int = 6,
) -> nn.Module:
    return EfficientFormerV2_Seg(in_channels=in_channels, num_classes=num_classes)