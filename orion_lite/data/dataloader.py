# -*- coding: utf-8 -*-
"""
DataLoader 构建器

提供统一的接口来构建训练/测试数据加载器。
"""

from typing import Tuple, Optional

import torch
from torch.utils.data import DataLoader

from .datasets import Market1501, build_market1501
from .transforms import build_transforms
from .samplers import RandomIdentitySampler, SequentialSampler


def build_dataloader(
    cfg,
    mode: str = 'train',
) -> DataLoader:
    """
    根据配置构建 DataLoader
    
    Args:
        cfg: 配置对象
        mode: 'train', 'query', 'gallery'
        
    Returns:
        DataLoader 实例
    """
    # 构建变换
    is_train = (mode == 'train')
    transform = build_transforms(
        height=cfg.DATA.HEIGHT,
        width=cfg.DATA.WIDTH,
        is_train=is_train,
        random_erasing_prob=cfg.DATA.AUGMENTATION.RE_PROB if is_train else 0.0,
        padding=cfg.DATA.AUGMENTATION.PADDING if is_train else 0,
        mean=tuple(cfg.DATA.NORM.MEAN),
        std=tuple(cfg.DATA.NORM.STD),
    )
    
    # 构建数据集
    dataset = build_market1501(
        root=cfg.DATA.ROOT,
        mode=mode,
        transform=transform,
    )
    
    # 构建采样器和 DataLoader
    if is_train:
        sampler = RandomIdentitySampler(
            data_source=dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            num_instances=cfg.TRAIN.SAMPLER.NUM_INSTANCES,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,  # MPS 不支持 pin_memory
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKERS,
            pin_memory=False,  # MPS 不支持 pin_memory
            drop_last=False,
        )
    
    return dataloader


def build_train_dataloader(
    root: str,
    height: int = 256,
    width: int = 128,
    batch_size: int = 64,
    num_instances: int = 4,
    num_workers: int = 4,
    random_erasing_prob: float = 0.5,
) -> Tuple[DataLoader, int]:
    """
    构建训练 DataLoader（简化接口）
    
    Args:
        root: 数据集根目录
        height: 图像高度
        width: 图像宽度
        batch_size: batch 大小
        num_instances: PK 采样中每个 ID 的图片数
        num_workers: 数据加载工作进程数
        random_erasing_prob: Random Erasing 概率
        
    Returns:
        (dataloader, num_classes): DataLoader 和类别数
    """
    # 构建变换
    transform = build_transforms(
        height=height,
        width=width,
        is_train=True,
        random_erasing_prob=random_erasing_prob,
    )
    
    # 构建数据集
    dataset = Market1501(
        root=root,
        mode='train',
        transform=transform,
    )
    
    # 构建采样器
    sampler = RandomIdentitySampler(
        data_source=dataset,
        batch_size=batch_size,
        num_instances=num_instances,
    )
    
    # 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,  # MPS 不支持
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return dataloader, dataset.num_classes


def build_test_dataloader(
    root: str,
    mode: str = 'query',
    height: int = 256,
    width: int = 128,
    batch_size: int = 128,
    num_workers: int = 4,
) -> DataLoader:
    """
    构建测试 DataLoader（简化接口）
    
    Args:
        root: 数据集根目录
        mode: 'query' 或 'gallery'
        height: 图像高度
        width: 图像宽度
        batch_size: batch 大小
        num_workers: 数据加载工作进程数
        
    Returns:
        DataLoader 实例
    """
    # 构建变换（无数据增强）
    transform = build_transforms(
        height=height,
        width=width,
        is_train=False,
    )
    
    # 构建数据集
    dataset = Market1501(
        root=root,
        mode=mode,
        transform=transform,
    )
    
    # 构建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # MPS 不支持
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    return dataloader


def collate_fn(batch):
    """
    自定义 collate 函数
    
    将 batch 中的字典列表整理成批次张量。
    
    Args:
        batch: [{'img': tensor, 'pid': int, 'camid': int, 'img_path': str}, ...]
        
    Returns:
        dict: {
            'img': (B, C, H, W),
            'pid': (B,),
            'camid': (B,),
            'img_path': [str, ...]
        }
    """
    imgs = torch.stack([item['img'] for item in batch])
    pids = torch.tensor([item['pid'] for item in batch])
    camids = torch.tensor([item['camid'] for item in batch])
    img_paths = [item['img_path'] for item in batch]
    
    return {
        'img': imgs,
        'pid': pids,
        'camid': camids,
        'img_path': img_paths,
    }


if __name__ == "__main__":
    # 测试代码
    import os
    
    data_root = "/Users/joseph/Desktop/【人工智能技术结课项目】ReID重识别/data/Market-1501"
    
    print("Testing DataLoader Builder...")
    
    if not os.path.exists(data_root):
        print(f"⚠️ 数据目录不存在: {data_root}")
        print("请先下载 Market-1501 数据集")
    else:
        # 测试训练 DataLoader
        train_loader, num_classes = build_train_dataloader(
            root=data_root,
            batch_size=32,
            num_instances=4,
            num_workers=0,  # macOS 建议设为 0
        )
        
        print(f"Training DataLoader:")
        print(f"  Num classes: {num_classes}")
        print(f"  Num batches: {len(train_loader)}")
        
        # 获取一个 batch
        batch = next(iter(train_loader))
        print(f"  Batch shape: {batch['img'].shape}")
        print(f"  PIDs: {batch['pid'][:8]}")
        
        print("✅ DataLoader test passed!")
