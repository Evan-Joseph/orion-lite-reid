# -*- coding: utf-8 -*-
"""
数据变换 (Data Transforms)

实现 Re-ID 任务常用的数据增强策略，参考 BoT (Bag of Tricks) 的最佳实践。

训练阶段变换：
1. Resize → 256×128
2. Random Horizontal Flip
3. Random Crop with Padding
4. ToTensor + Normalize
5. Random Erasing (模拟遮挡)

测试阶段变换：
1. Resize → 256×128
2. ToTensor + Normalize
"""

import math
import random
from typing import Tuple, List

import torch
from torchvision import transforms as T


class RandomErasing:
    """
    Random Erasing 数据增强
    
    论文: Random Erasing Data Augmentation (AAAI 2020)
    
    在图像的随机区域填充随机值，模拟遮挡情况。
    
    Args:
        probability (float): 应用的概率，默认 0.5
        sl (float): 擦除区域面积下限（相对于图像面积）
        sh (float): 擦除区域面积上限
        r1 (float): 擦除区域长宽比下限
        mean (tuple): 填充的均值 (ImageNet mean)
    """
    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.4,
        r1: float = 0.3,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    ):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: 图像张量 (C, H, W)
            
        Returns:
            可能被擦除的图像张量
        """
        if random.random() > self.probability:
            return img

        c, h, w = img.shape
        area = h * w

        for _ in range(100):  # 最多尝试 100 次
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            eh = int(round(math.sqrt(target_area * aspect_ratio)))
            ew = int(round(math.sqrt(target_area / aspect_ratio)))

            if eh < h and ew < w:
                x1 = random.randint(0, h - eh)
                y1 = random.randint(0, w - ew)
                
                # 使用 ImageNet mean 填充
                img[0, x1:x1+eh, y1:y1+ew] = self.mean[0]
                img[1, x1:x1+eh, y1:y1+ew] = self.mean[1]
                img[2, x1:x1+eh, y1:y1+ew] = self.mean[2]
                return img

        return img


def build_transforms(
    height: int = 256,
    width: int = 128,
    is_train: bool = True,
    random_erasing_prob: float = 0.5,
    padding: int = 10,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> T.Compose:
    """
    构建数据变换流水线
    
    Args:
        height (int): 目标图像高度
        width (int): 目标图像宽度
        is_train (bool): 是否为训练模式
        random_erasing_prob (float): Random Erasing 的概率
        padding (int): Random Crop 时的 padding 大小
        mean (tuple): 归一化均值 (ImageNet)
        std (tuple): 归一化标准差 (ImageNet)
        
    Returns:
        transforms.Compose: 变换流水线
    """
    if is_train:
        # 训练阶段变换
        transform = T.Compose([
            # 1. Resize
            T.Resize((height, width)),
            
            # 2. Random Horizontal Flip (概率 0.5)
            T.RandomHorizontalFlip(p=0.5),
            
            # 3. Random Crop with Padding (增加位移不变性)
            T.Pad(padding),
            T.RandomCrop((height, width)),
            
            # 4. ToTensor + Normalize
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            
            # 5. Random Erasing (模拟遮挡)
            RandomErasing(probability=random_erasing_prob, mean=mean),
        ])
    else:
        # 测试阶段变换（无数据增强）
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    return transform


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    反归一化，用于可视化
    
    Args:
        tensor: 归一化后的图像张量 (C, H, W) 或 (B, C, H, W)
        mean: 归一化均值
        std: 归一化标准差
        
    Returns:
        反归一化后的图像张量，值域 [0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


if __name__ == "__main__":
    # 测试代码
    from PIL import Image
    import numpy as np
    
    print("Testing Data Transforms...")
    
    # 创建测试图像
    img = Image.new('RGB', (100, 200), color='red')
    
    # 测试训练变换
    train_transform = build_transforms(is_train=True)
    train_out = train_transform(img)
    print(f"Train Transform Output: {train_out.shape}")
    assert train_out.shape == (3, 256, 128), "错误的输出尺寸"
    
    # 测试测试变换
    test_transform = build_transforms(is_train=False)
    test_out = test_transform(img)
    print(f"Test Transform Output: {test_out.shape}")
    assert test_out.shape == (3, 256, 128), "错误的输出尺寸"
    
    # 测试 Random Erasing
    random_erasing = RandomErasing(probability=1.0)  # 强制触发
    erased = random_erasing(train_out.clone())
    print(f"Random Erasing applied: {not torch.equal(train_out, erased)}")
    
    print("✅ All transforms tests passed!")
