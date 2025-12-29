# -*- coding: utf-8 -*-
"""
评估器 (Evaluator)

实现 Re-ID 的标准评估流程：
1. 提取 Query 和 Gallery 特征
2. 计算相似度矩阵
3. 计算 CMC 和 mAP 指标
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    """
    Re-ID 评估器
    
    Args:
        model: 模型
        device: 评估设备
        feat_norm: 是否对特征进行 L2 归一化
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        feat_norm: bool = True,
    ):
        self.model = model
        self.device = device
        self.feat_norm = feat_norm
    
    @torch.no_grad()
    def extract_features(self, dataloader: DataLoader) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        提取数据集的特征
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            features: 特征矩阵 (N, D)
            pids: 身份标签 (N,)
            camids: 摄像头标签 (N,)
        """
        self.model.eval()
        
        features_list = []
        pids_list = []
        camids_list = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            imgs = batch['img'].to(self.device)
            pids = batch['pid'].numpy()
            camids = batch['camid'].numpy()
            
            # 提取特征
            feats = self.model(imgs)
            
            # L2 归一化
            if self.feat_norm:
                feats = F.normalize(feats, p=2, dim=1)
            
            features_list.append(feats.cpu())
            pids_list.append(pids)
            camids_list.append(camids)
        
        features = torch.cat(features_list, dim=0)
        pids = np.concatenate(pids_list)
        camids = np.concatenate(camids_list)
        
        return features, pids, camids
    
    def evaluate(
        self,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        max_rank: int = 50,
    ) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            query_loader: Query 数据加载器
            gallery_loader: Gallery 数据加载器
            max_rank: 最大 rank 值
            
        Returns:
            metrics: 评估指标字典
        """
        print("📊 Extracting Query features...")
        q_feats, q_pids, q_camids = self.extract_features(query_loader)
        
        print("📊 Extracting Gallery features...")
        g_feats, g_pids, g_camids = self.extract_features(gallery_loader)
        
        print("🔍 Computing distance matrix...")
        # 使用余弦相似度（因为已经 L2 归一化，所以直接矩阵乘法）
        distmat = 1 - torch.mm(q_feats, g_feats.t()).numpy()
        
        print("📈 Computing CMC and mAP...")
        cmc, mAP = self.compute_metrics(
            distmat=distmat,
            q_pids=q_pids,
            g_pids=g_pids,
            q_camids=q_camids,
            g_camids=g_camids,
            max_rank=max_rank,
        )
        
        metrics = {
            'Rank-1': cmc[0] * 100,
            'Rank-5': cmc[4] * 100,
            'Rank-10': cmc[9] * 100,
            'mAP': mAP * 100,
        }
        
        # 打印结果
        print("=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for name, value in metrics.items():
            print(f"  {name}: {value:.2f}%")
        print("=" * 50)
        
        return metrics
    
    def compute_metrics(
        self,
        distmat: np.ndarray,
        q_pids: np.ndarray,
        g_pids: np.ndarray,
        q_camids: np.ndarray,
        g_camids: np.ndarray,
        max_rank: int = 50,
    ) -> Tuple[np.ndarray, float]:
        """
        计算 CMC 和 mAP
        
        Market-1501 协议：
        - 排除与 query 相同摄像头相同身份的 gallery 样本
        - 排除 junk 图像（pid = -1 或 0）
        
        Args:
            distmat: 距离矩阵 (num_query, num_gallery)
            q_pids: Query 身份标签
            g_pids: Gallery 身份标签
            q_camids: Query 摄像头标签
            g_camids: Gallery 摄像头标签
            max_rank: 最大 rank 值
            
        Returns:
            cmc: CMC 曲线
            mAP: 平均精确度
        """
        num_query, num_gallery = distmat.shape
        
        if num_gallery < max_rank:
            max_rank = num_gallery
            print(f"⚠️ Gallery size ({num_gallery}) < max_rank, adjusted max_rank to {max_rank}")
        
        # 排序索引
        indices = np.argsort(distmat, axis=1)
        
        # 初始化
        all_cmc = []
        all_AP = []
        num_valid_query = 0
        
        for q_idx in range(num_query):
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]
            
            # 排序后的 gallery
            order = indices[q_idx]
            
            # 创建 mask：排除相同摄像头相同身份 AND junk 图像
            remove_mask = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            junk_mask = (g_pids[order] < 0) | (g_pids[order] == 0)
            keep_mask = ~(remove_mask | junk_mask)
            
            # 获取匹配结果
            matches = (g_pids[order] == q_pid) & keep_mask
            
            if not np.any(matches):
                # 没有有效的正样本，跳过
                continue
            
            num_valid_query += 1
            
            # 过滤后的匹配
            matches_filtered = matches[keep_mask]
            
            # 计算 CMC
            cmc = matches_filtered.cumsum()
            cmc[cmc > 1] = 1
            
            # 补齐到 max_rank
            if len(cmc) < max_rank:
                cmc = np.pad(cmc, (0, max_rank - len(cmc)), constant_values=cmc[-1] if len(cmc) > 0 else 0)
            else:
                cmc = cmc[:max_rank]
            
            all_cmc.append(cmc)
            
            # 计算 AP
            num_rel = matches_filtered.sum()
            tmp_cmc = matches_filtered.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * matches_filtered
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        
        if num_valid_query == 0:
            print("⚠️ No valid query found!")
            return np.zeros(max_rank), 0.0
        
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        cmc = all_cmc.mean(axis=0)
        mAP = np.mean(all_AP)
        
        return cmc, mAP


if __name__ == "__main__":
    # 简单测试
    print("Testing Evaluator...")
    
    # 模拟数据
    num_query = 100
    num_gallery = 500
    feat_dim = 512
    num_classes = 50
    
    # 随机距离矩阵
    distmat = np.random.rand(num_query, num_gallery)
    
    # 随机标签
    q_pids = np.random.randint(0, num_classes, num_query)
    g_pids = np.random.randint(0, num_classes, num_gallery)
    q_camids = np.random.randint(1, 7, num_query)
    g_camids = np.random.randint(1, 7, num_gallery)
    
    # 测试评估计算
    evaluator = Evaluator(model=None, device=torch.device('cpu'))
    cmc, mAP = evaluator.compute_metrics(
        distmat=distmat,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        max_rank=50,
    )
    
    print(f"Rank-1: {cmc[0]*100:.2f}%")
    print(f"mAP: {mAP*100:.2f}%")
    print("✅ Evaluator test passed!")
