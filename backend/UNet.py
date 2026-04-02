"""
U-Net++ 模型 - 适配4通道遥感影像语义分割
【修复版】已移除多余的上采样操作，确保输入输出尺寸一致
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """卷积块：两个3x3卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetPlusPlus(nn.Module):
    """
    U-Net++ 模型 - 简化实现 (标准版)
    输入: (B, 4, 512, 512)
    输出: (B, 6, 512, 512)
    """
    def __init__(self, in_channels=4, num_classes=6, features=64):
        super().__init__()
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.enc4 = ConvBlock(features * 4, features * 8)
        self.enc5 = ConvBlock(features * 8, features * 16)
        
        # 池化
        self.pool = nn.MaxPool2d(2)
        
        # 解码器 - 标准U-Net结构
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.dec4 = ConvBlock(features * 16, features * 8)
        
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = ConvBlock(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        
        # 输出层
        self.output = nn.Conv2d(features, num_classes, 1)
        
    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        
        # 解码器
        d4 = self.up4(e5)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 输出 (已通过转置卷积恢复到 512x512，无需再采样)
        out = self.output(d1)
        
        return out


class UNetPlusPlusNested(nn.Module):
    """
    U-Net++ 模型 - 完整嵌套版本
    使用显式的层定义，避免索引问题
    """
    def __init__(self, in_channels=4, num_classes=6, features=64):
        super().__init__()
        
        # ========== 编码器 ==========
        # 第0层
        self.x00 = ConvBlock(in_channels, features)
        
        # 第1层
        self.x10 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(features, features * 2)
        )
        
        # 第2层
        self.x20 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(features * 2, features * 4)
        )
        
        # 第3层
        self.x30 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(features * 4, features * 8)
        )
        
        # 第4层
        self.x40 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(features * 8, features * 16)
        )
        
        # ========== 解码器 - 嵌套连接 ==========
        # x01
        self.up_x10 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x01 = ConvBlock(features + features * 2, features)
        
        # x11
        self.up_x20 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x11 = ConvBlock(features * 2 + features * 4, features * 2)
        
        # x02
        self.up_x11 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x02 = ConvBlock(features + features * 2, features)
        
        # x21
        self.up_x30 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x21 = ConvBlock(features * 4 + features * 8, features * 4)
        
        # x12
        self.up_x21 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x12 = ConvBlock(features * 2 + features * 4, features * 2)
        
        # x03
        self.up_x12 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x03 = ConvBlock(features + features * 2, features)
        
        # x31
        self.up_x40 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x31 = ConvBlock(features * 8 + features * 16, features * 8)
        
        # x22
        self.up_x31 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x22 = ConvBlock(features * 4 + features * 8, features * 4)
        
        # x13
        self.up_x22 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x13 = ConvBlock(features * 2 + features * 4, features * 2)
        
        # x04
        self.up_x13 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.x04 = ConvBlock(features + features * 2, features)
        
        # ========== 输出层 ==========
        self.final_conv = nn.Conv2d(features, num_classes, 1)
        
    def forward(self, x):
        # 编码器
        x00 = self.x00(x)
        x10 = self.x10(x00)
        x20 = self.x20(x10)
        x30 = self.x30(x20)
        x40 = self.x40(x30)
        
        # 解码器 - 嵌套连接
        # 第1层嵌套
        x01 = self.x01(torch.cat([x00, self.up_x10(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up_x20(x20)], dim=1))
        x02 = self.x02(torch.cat([x00, self.up_x11(x11)], dim=1))
        
        # 第2层嵌套
        x21 = self.x21(torch.cat([x20, self.up_x30(x30)], dim=1))
        x12 = self.x12(torch.cat([x10, self.up_x21(x21)], dim=1))
        x03 = self.x03(torch.cat([x00, self.up_x12(x12)], dim=1))
        
        # 第3层嵌套
        x31 = self.x31(torch.cat([x30, self.up_x40(x40)], dim=1))
        x22 = self.x22(torch.cat([x20, self.up_x31(x31)], dim=1))
        x13 = self.x13(torch.cat([x10, self.up_x22(x22)], dim=1))
        x04 = self.x04(torch.cat([x00, self.up_x13(x13)], dim=1))
        
        # 输出（使用最深层）
        out = self.final_conv(x04)
        
        return out


class UNetPlusPlusLight(nn.Module):
    """
    轻量级U-Net++模型 - 4层深度
    【修复版】移除了最后的多余上采样
    """
    def __init__(self, in_channels=4, num_classes=6, features=32):
        super().__init__()
        
        # 编码器
        self.enc1 = ConvBlock(in_channels, features)
        self.enc2 = ConvBlock(features, features * 2)
        self.enc3 = ConvBlock(features * 2, features * 4)
        self.enc4 = ConvBlock(features * 4, features * 8)
        
        self.pool = nn.MaxPool2d(2)
        
        # 解码器
        self.up3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.dec3 = ConvBlock(features * 8, features * 4)
        
        self.up2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.dec2 = ConvBlock(features * 4, features * 2)
        
        self.up1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.dec1 = ConvBlock(features * 2, features)
        
        # 输出
        self.output = nn.Conv2d(features, num_classes, 1)
        
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # 解码
        d3 = self.up3(e4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # 输出 (d1 已经是 512x512，直接输出)
        out = self.output(d1)
        
        return out


def create_unetplusplus(model_type="standard", in_channels=4, num_classes=6):
    """
    创建U-Net++模型
    
    Args:
        model_type: 'standard', 'nested', 'light', 'small', 'large'
        in_channels: 输入通道数
        num_classes: 输出类别数
    
    Returns:
        U-Net++模型实例
    """
    if model_type == "standard":
        return UNetPlusPlus(in_channels, num_classes, features=64)
    elif model_type == "nested":
        return UNetPlusPlusNested(in_channels, num_classes, features=64)
    elif model_type == "light":
        return UNetPlusPlusLight(in_channels, num_classes, features=32)
    elif model_type == "small":
        return UNetPlusPlus(in_channels, num_classes, features=32)
    elif model_type == "large":
        return UNetPlusPlus(in_channels, num_classes, features=128)
    else:
        return UNetPlusPlus(in_channels, num_classes, features=64)






