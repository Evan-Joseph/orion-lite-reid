# -*- coding: utf-8 -*-
"""
PK Sampler (Identity Sampler)

Re-ID 训练中常用的采样策略，确保每个 batch 中包含 P 个不同的身份，
每个身份有 K 张图片。这对于 Triplet Loss / Circle Loss 等度量学习损失
至关重要，因为它们需要正负样本对。

例如：P=16, K=4 → batch_size = 64，包含 16 个不同的人，每人 4 张图
"""

import copy
import random
from collections import defaultdict
from typing import Iterator, List, Dict, Optional

from torch.utils.data import Sampler


class RandomIdentitySampler(Sampler):
    """
    Random Identity Sampler (PK Sampler)
    
    确保每个 mini-batch 包含 P 个随机选择的身份，每个身份有 K 张图片。
    
    Args:
        data_source: 数据集，需要有 get_pid_to_indices() 方法
        batch_size (int): batch 大小，应该等于 P * K
        num_instances (int): 每个身份的图片数 K
    """
    def __init__(
        self,
        data_source,
        batch_size: int,
        num_instances: int = 4,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        
        # P = batch_size / K
        self.num_pids_per_batch = batch_size // num_instances
        
        # 获取每个 PID 对应的样本索引
        self.pid_to_indices = data_source.get_pid_to_indices()
        self.pids = list(self.pid_to_indices.keys())
        
        # 计算 epoch 长度
        # 确保每个 PID 至少被采样一次
        self.length = 0
        for pid in self.pids:
            num_samples = len(self.pid_to_indices[pid])
            self.length += max(num_samples, self.num_instances)
    
    def __iter__(self) -> Iterator[int]:
        """
        生成采样索引
        
        采样策略：
        1. 随机打乱所有 PID
        2. 对于每个 PID，随机选择 K 张图片
        3. 如果某个 PID 的图片数 < K，则重复采样
        """
        # 为每个 PID 创建索引副本
        pid_to_indices = copy.deepcopy(self.pid_to_indices)
        
        # 为每个 PID 的索引列表打乱顺序
        for pid in pid_to_indices:
            random.shuffle(pid_to_indices[pid])
        
        # 收集所有需要返回的索引
        indices = []
        
        # 随机打乱 PID 顺序
        pids = copy.deepcopy(self.pids)
        random.shuffle(pids)
        
        # 为每个 PID 生成采样索引
        pid_indices_dict = defaultdict(list)
        for pid in pids:
            pid_samples = pid_to_indices[pid]
            num_samples = len(pid_samples)
            
            if num_samples >= self.num_instances:
                # 如果图片数 >= K，随机选 K 张
                selected = random.sample(pid_samples, self.num_instances)
            else:
                # 如果图片数 < K，重复采样补齐
                selected = pid_samples * (self.num_instances // num_samples + 1)
                selected = selected[:self.num_instances]
            
            pid_indices_dict[pid] = selected
        
        # 组织成 batch
        # 每次取 P 个 PID，每个 PID 取 K 张
        pids_shuffled = list(pid_indices_dict.keys())
        random.shuffle(pids_shuffled)
        
        batch = []
        for pid in pids_shuffled:
            batch.extend(pid_indices_dict[pid])
            
            # 当 batch 达到目标大小时，yield 并重置
            if len(batch) >= self.batch_size:
                indices.extend(batch[:self.batch_size])
                batch = batch[self.batch_size:]
        
        # 处理剩余的样本（可能不足一个完整 batch）
        if batch:
            indices.extend(batch)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """返回采样的总样本数"""
        # 向下取整到 batch_size 的整数倍
        return (self.length // self.batch_size) * self.batch_size


class RandomSampler(Sampler):
    """
    简单随机采样器
    
    用于测试阶段或不需要 PK 采样的场景。
    """
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        random.shuffle(indices)
        return iter(indices)
    
    def __len__(self) -> int:
        return len(self.data_source)


class SequentialSampler(Sampler):
    """
    顺序采样器
    
    用于测试/评估阶段，确保结果可复现。
    """
    def __init__(self, data_source):
        self.data_source = data_source
    
    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))
    
    def __len__(self) -> int:
        return len(self.data_source)


if __name__ == "__main__":
    # 测试代码
    print("Testing PK Sampler...")
    
    # 模拟一个简单的数据集
    class MockDataset:
        def __init__(self):
            # 模拟 10 个 PID，每个 PID 有不同数量的图片
            self.samples = []
            pid_counts = [5, 3, 8, 2, 6, 4, 7, 1, 9, 4]
            
            idx = 0
            for pid, count in enumerate(pid_counts):
                for _ in range(count):
                    self.samples.append((f"img_{idx}.jpg", pid, 1))
                    idx += 1
        
        def __len__(self):
            return len(self.samples)
        
        def get_pid_to_indices(self):
            pid_to_indices = defaultdict(list)
            for idx, (_, pid, _) in enumerate(self.samples):
                pid_to_indices[pid].append(idx)
            return dict(pid_to_indices)
    
    # 创建模拟数据集
    dataset = MockDataset()
    print(f"Dataset size: {len(dataset)}")
    
    # 测试 PK Sampler
    sampler = RandomIdentitySampler(
        data_source=dataset,
        batch_size=16,  # P=4, K=4
        num_instances=4,
    )
    
    # 采样一个 epoch
    indices = list(sampler)
    print(f"Sampled indices count: {len(indices)}")
    
    # 验证每个 batch 的 PID 分布
    batch_size = 16
    for i in range(0, min(32, len(indices)), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_pids = [dataset.samples[idx][1] for idx in batch_indices]
        unique_pids = set(batch_pids)
        print(f"Batch {i//batch_size}: {len(unique_pids)} unique PIDs, counts: {[batch_pids.count(p) for p in unique_pids]}")
    
    print("✅ PK Sampler test passed!")
