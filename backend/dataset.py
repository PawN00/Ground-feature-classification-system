import os
import random
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# ===================== 核心配置 =====================
# 严格对应 GID-5 标签颜色
RGB_TO_LABEL = {
    0: (0, 0, 0),  # unlabeled
    1: (255, 0, 0),  # built-up
    2: (0, 255, 0),  # farmland
    3: (0, 255, 255),  # forest (注意：你之前图中forest是青色 0,255,255)
    4: (255, 255, 0),  # meadow
    5: (0, 0, 255)  # water
}


# ===================== 工具函数 =====================
def rgb2label(rgb_label_data):
    """将 RGB 标签转换为单通道 Class ID"""
    # rasterio read: (C, H, W) -> (H, W, C)
    if rgb_label_data.shape[0] == 3:
        rgb_label_data = np.transpose(rgb_label_data, (1, 2, 0))

    h, w, _ = rgb_label_data.shape
    label_data = np.zeros((h, w), dtype=np.int64)
    for label_val, (r, g, b) in RGB_TO_LABEL.items():
        mask = (rgb_label_data[..., 0] == r) & \
               (rgb_label_data[..., 1] == g) & \
               (rgb_label_data[..., 2] == b)
        label_data[mask] = label_val
    return label_data


# ===================== 增强类 =====================
class SegTransform:
    """语义分割同步变换"""

    def __init__(self, size=(256, 256), is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img, mask):
        # 这里的 img 是 (C, H, W) Tensor, mask 是 (H, W) LongTensor
        if self.is_train:
            # 随机水平翻转
            if random.random() > 0.5:
                img = torch.flip(img, dims=[2])
                mask = torch.flip(mask, dims=[1])
            # 随机垂直翻转
            if random.random() > 0.5:
                img = torch.flip(img, dims=[1])
                mask = torch.flip(mask, dims=[0])
            # 随机 90 度旋转
            if random.random() > 0.5:
                k = random.randint(1, 3)
                img = torch.rot90(img, k, dims=[1, 2])
                mask = torch.rot90(mask, k, dims=[0, 1])

        # 插值缩放
        # img: [C, H, W] -> [1, C, H, W] for interpolate
        img = torch.nn.functional.interpolate(img.unsqueeze(0), size=self.size, mode='bilinear',
                                              align_corners=False).squeeze(0)
        # mask: [H, W] -> [1, 1, H, W] for interpolate
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=self.size,
                                               mode='nearest').squeeze(0).squeeze(0).long()

        return img, mask


# ===================== 数据集类 =====================
class RemoteSensingTxtDataset(Dataset):
    """基于 TXT 列表加载影像和标签"""

    def __init__(self, txt_path, size=(256, 256), is_train=True):
        self.size = size
        self.is_train = is_train
        self.transform = SegTransform(size, is_train)
        self.samples = []

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"❌ 找不到列表文件: {txt_path}")

        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    self.samples.append((parts[0], parts[1]))

        print(f"📊 已从 {os.path.basename(txt_path)} 加载样本数量：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # 1. 读取影像
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)
            # 基础归一化 (Min-Max)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 2. 读取标签并转换
        with rasterio.open(label_path) as src:
            rgb_label = src.read()
            label = rgb2label(rgb_label)

        # 3. 转为 Tensor
        img_tensor = torch.from_numpy(img).float()
        label_tensor = torch.from_numpy(label).long()

        # 4. 同步增强与缩放
        img_tensor, label_tensor = self.transform(img_tensor, label_tensor)

        return img_tensor, label_tensor


# ===================== 构建加载器函数 =====================
def get_dataloaders(root_dir, batch_size=16, img_size=(256, 256)):
    """
    root_dir: 存放 train_list.txt 等文件的目录
    """
    train_txt = os.path.join(root_dir, 'train_list.txt')
    valid_txt = os.path.join(root_dir, 'valid_list.txt')
    test_txt = os.path.join(root_dir, 'test_list.txt')

    train_ds = RemoteSensingTxtDataset(train_txt, size=img_size, is_train=True)
    valid_ds = RemoteSensingTxtDataset(valid_txt, size=img_size, is_train=False)
    test_ds = RemoteSensingTxtDataset(test_txt, size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader


# ===================== 测试运行 =====================
if __name__ == "__main__":
    # 替换为你 txt 文件所在的实际目录
    LIST_ROOT = r'D:\GID5\ershui_jieya\GID5'

    try:
        train_loader, val_loader, test_loader = get_dataloaders(LIST_ROOT, batch_size=8)

        # 尝试取出一个 Batch 验证
        imgs, labels = next(iter(train_loader))
        print(f"\n✅ 成功加载 Batch!")
        print(f"影像尺寸: {imgs.shape}")  # [Batch, Channel, H, W]
        print(f"标签尺寸: {labels.shape}")  # [Batch, H, W]
        print(f"标签类别范围: {torch.unique(labels)}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")