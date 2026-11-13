# -*- coding: utf-8 -*-
"""
Market-1501 数据集

Market-1501 数据集由清华大学发布，是行人重识别领域最经典的基准数据集。
包含 6 个摄像头拍摄的 1501 个行人身份。

数据集结构：
- bounding_box_train/: 训练集 (12,936 张, 751 个ID)
- bounding_box_test/: 测试集/Gallery (19,732 张, 750 个ID)
- query/: 查询集/Probe (3,368 张)

文件命名格式：0001_c1s1_001051_00.jpg
- 0001: Person ID (身份标签)
- c1: Camera ID (摄像头编号, c1-c6)
- s1: Sequence ID (视频序列编号)
- 001051: Frame ID (帧号)
- 00: Bounding Box ID
"""

import os
import re
from PIL import Image
from typing import List, Tuple, Dict, Optional, Callable
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class Market1501(Dataset):
    """
    Market-1501 数据集
    
    Args:
        root (str): 数据集根目录，应包含 bounding_box_train, bounding_box_test, query 等子目录
        mode (str): 数据集模式
            - 'train': 训练集 (bounding_box_train)
            - 'query': 查询集 (query)
            - 'gallery': 图库集 (bounding_box_test)
        transform (callable): 图像变换
        verbose (bool): 是否打印数据集信息
    """
    
    # 类属性：子目录映射
    SUBDIR_MAP = {
        'train': 'bounding_box_train',
        'query': 'query',
        'gallery': 'bounding_box_test',
    }
    
    def __init__(
        self,
        root: str,
        mode: str = 'train',
        transform: Optional[Callable] = None,
        verbose: bool = True,
    ):
        super(Market1501, self).__init__()
        
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # 验证模式
        assert mode in self.SUBDIR_MAP, f"Invalid mode: {mode}. Choose from {list(self.SUBDIR_MAP.keys())}"
        
        # 构建数据目录路径
        self.data_dir = os.path.join(root, self.SUBDIR_MAP[mode])
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"数据目录不存在: {self.data_dir}\n"
                f"请确保 Market-1501 数据集已解压到: {root}"
            )
        
        # 加载数据
        self.samples: List[Tuple[str, int, int]] = []  # (img_path, pid, camid)
        self.pid2label: Dict[int, int] = {}  # 原始 PID -> 连续标签
        self._load_data()
        
        if verbose:
            self._print_info()
    
    def _load_data(self):
        """加载数据集"""
        # 收集所有图像
        img_paths = []
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.jpg'):
                continue
            # 过滤掉干扰项 (pid = -1 或 0)
            pid = int(filename.split('_')[0])
            if pid <= 0:
                continue
            img_paths.append(os.path.join(self.data_dir, filename))
        
        # 解析文件名并构建样本列表
        pid_container = set()
        for img_path in img_paths:
            pid, camid = self._parse_filename(os.path.basename(img_path))
            if pid == -1:
                continue
            pid_container.add(pid)
            self.samples.append((img_path, pid, camid))
        
        # 构建 PID -> 连续标签映射（仅训练集需要）
        if self.mode == 'train':
            pid_list = sorted(list(pid_container))
            self.pid2label = {pid: label for label, pid in enumerate(pid_list)}
            # 更新样本列表，将原始 PID 替换为连续标签
            self.samples = [
                (path, self.pid2label[pid], camid) 
                for path, pid, camid in self.samples
            ]
        else:
            # 测试阶段保留原始 PID
            self.pid2label = {pid: pid for pid in pid_container}
    
    def _parse_filename(self, filename: str) -> Tuple[int, int]:
        """
        解析 Market-1501 文件名
        
        格式: 0001_c1s1_001051_00.jpg
        
        Returns:
            (pid, camid): 行人ID, 摄像头ID
        """
        # 正则匹配
        pattern = r'(\d+)_c(\d+)s(\d+)_(\d+)_(\d+)\.jpg'
        match = re.match(pattern, filename)
        
        if match:
            pid = int(match.group(1))
            camid = int(match.group(2))
            return pid, camid
        else:
            # 尝试简单解析
            try:
                parts = filename.split('_')
                pid = int(parts[0])
                camid = int(parts[1][1])  # 取 'c1' 中的 '1'
                return pid, camid
            except:
                return -1, -1
    
    def _print_info(self):
        """打印数据集信息"""
        num_pids = len(set(pid for _, pid, _ in self.samples))
        num_cams = len(set(camid for _, _, camid in self.samples))
        
        print(f"=" * 50)
        print(f"Market-1501 Dataset ({self.mode})")
        print(f"=" * 50)
        print(f"  Directory: {self.data_dir}")
        print(f"  Images:    {len(self.samples)}")
        print(f"  IDs:       {num_pids}")
        print(f"  Cameras:   {num_cams}")
        print(f"=" * 50)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict:
        """
        获取样本
        
        Returns:
            dict: {
                'img': 图像张量 (C, H, W),
                'pid': 行人ID (训练集为连续标签, 测试集为原始ID),
                'camid': 摄像头ID,
                'img_path': 图像路径 (用于调试)
            }
        """
        img_path, pid, camid = self.samples[index]
        
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            img = self.transform(img)
        
        return {
            'img': img,
            'pid': pid,
            'camid': camid,
            'img_path': img_path,
        }
    
    @property
    def num_classes(self) -> int:
        """返回类别数（身份数）"""
        return len(set(pid for _, pid, _ in self.samples))
    
    @property
    def num_cameras(self) -> int:
        """返回摄像头数"""
        return len(set(camid for _, _, camid in self.samples))
    
    def get_pid_to_indices(self) -> Dict[int, List[int]]:
        """
        获取每个 PID 对应的样本索引列表
        用于 PK Sampler
        
        Returns:
            dict: {pid: [idx1, idx2, ...]}
        """
        pid_to_indices = defaultdict(list)
        for idx, (_, pid, _) in enumerate(self.samples):
            pid_to_indices[pid].append(idx)
        return dict(pid_to_indices)


def build_market1501(root: str, mode: str, transform=None) -> Market1501:
    """
    构建 Market-1501 数据集
    
    Args:
        root: 数据集根目录
        mode: 'train', 'query', 'gallery'
        transform: 图像变换
        
    Returns:
        Market1501 数据集实例
    """
    return Market1501(root=root, mode=mode, transform=transform)


if __name__ == "__main__":
    # 测试代码
    import sys
    
    # 假设数据集路径
    from pathlib import Path as _Path
    data_root = str(_Path(__file__).resolve().parents[2] / "datasets" / "Market-1501-v15.09.15")
    
    print("Testing Market-1501 Dataset...")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_root):
        print(f"⚠️ 数据目录不存在: {data_root}")
        print("请先下载 Market-1501 数据集")
        sys.exit(0)
    
    # 测试训练集
    try:
        train_dataset = Market1501(root=data_root, mode='train')
        print(f"\n样本示例:")
        sample = train_dataset[0]
        print(f"  PID: {sample['pid']}")
        print(f"  CamID: {sample['camid']}")
        print(f"  Path: {sample['img_path']}")
        print("✅ 训练集加载成功!")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
    
    # 测试查询集
    try:
        query_dataset = Market1501(root=data_root, mode='query')
        print("✅ 查询集加载成功!")
    except FileNotFoundError as e:
        print(f"⚠️ {e}")
