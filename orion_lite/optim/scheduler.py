# -*- coding: utf-8 -*-
"""
学习率调度器

实现 Warmup + Cosine Annealing 学习率策略，这是 Re-ID 训练中常用的调度方案。
"""

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineScheduler(LRScheduler):
    """
    Warmup + Cosine Annealing 学习率调度器
    
    学习率变化：
    1. Warmup 阶段：从 base_lr * warmup_factor 线性增加到 base_lr
    2. Cosine 阶段：按余弦曲线从 base_lr 下降到 min_lr
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup 的 epoch 数
        total_epochs: 总 epoch 数
        warmup_factor: warmup 起始学习率因子
        min_lr: 最小学习率
        last_epoch: 上一个 epoch（用于恢复训练）
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_factor: float = 0.01,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加
            alpha = self.last_epoch / self.warmup_epochs
            factor = self.warmup_factor + (1 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine 阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor 
                for base_lr in self.base_lrs
            ]


class WarmupMultiStepScheduler(LRScheduler):
    """
    Warmup + MultiStep 学习率调度器
    
    学习率变化：
    1. Warmup 阶段：线性增加
    2. MultiStep 阶段：在指定 epoch 处乘以 gamma
    
    Args:
        optimizer: 优化器
        warmup_epochs: warmup 的 epoch 数
        milestones: 学习率下降的 epoch 列表
        gamma: 学习率衰减因子
        warmup_factor: warmup 起始学习率因子
        last_epoch: 上一个 epoch
    """
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        
        super(WarmupMultiStepScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # Warmup 阶段
            alpha = self.last_epoch / self.warmup_epochs
            factor = self.warmup_factor + (1 - self.warmup_factor) * alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # MultiStep 阶段
            factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    total_epochs: int,
    warmup_epochs: int = 10,
    warmup_factor: float = 0.01,
    **kwargs
) -> LRScheduler:
    """
    构建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('warmup_cosine' 或 'warmup_multistep')
        total_epochs: 总 epoch 数
        warmup_epochs: warmup 的 epoch 数
        warmup_factor: warmup 起始学习率因子
        **kwargs: 其他参数
        
    Returns:
        学习率调度器实例
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            warmup_factor=warmup_factor,
            min_lr=kwargs.get('min_lr', 1e-7),
        )
    elif scheduler_name == 'warmup_multistep':
        milestones = kwargs.get('milestones', [30, 50])
        return WarmupMultiStepScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            milestones=milestones,
            gamma=kwargs.get('gamma', 0.1),
            warmup_factor=warmup_factor,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


if __name__ == "__main__":
    # 测试代码
    import torch
    import matplotlib.pyplot as plt
    
    print("Testing Schedulers...")
    
    # 创建模拟优化器
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00035)
    
    total_epochs = 60
    warmup_epochs = 10
    
    # 测试 Warmup Cosine
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        warmup_factor=0.01,
    )
    
    lrs = []
    for epoch in range(total_epochs):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print(f"Warmup Cosine: LR range [{min(lrs):.6f}, {max(lrs):.6f}]")
    print(f"  Epoch 0: {lrs[0]:.6f}")
    print(f"  Epoch {warmup_epochs}: {lrs[warmup_epochs]:.6f}")
    print(f"  Epoch {total_epochs-1}: {lrs[-1]:.6f}")
    
    print("✅ Scheduler test passed!")
