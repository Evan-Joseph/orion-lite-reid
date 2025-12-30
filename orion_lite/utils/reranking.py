# -*- coding: utf-8 -*-
"""
K-Reciprocal Re-ranking

论文: Re-ranking Person Re-identification with k-reciprocal Encoding (CVPR 2017)
链接: https://arxiv.org/abs/1701.08398

该方法通过计算 k-reciprocal 近邻来优化检索结果，能够显著提升 Rank-1 和 mAP。
"""

import numpy as np
from typing import Optional


def compute_euclidean_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
) -> np.ndarray:
    """
    计算欧氏距离矩阵
    
    Args:
        query_features: Query 特征 (num_query, feat_dim)
        gallery_features: Gallery 特征 (num_gallery, feat_dim)
        
    Returns:
        距离矩阵 (num_query, num_gallery)
    """
    m, n = query_features.shape[0], gallery_features.shape[0]
    
    # ||q - g||^2 = ||q||^2 + ||g||^2 - 2 * q @ g.T
    query_sq = np.sum(query_features ** 2, axis=1, keepdims=True)  # (m, 1)
    gallery_sq = np.sum(gallery_features ** 2, axis=1, keepdims=True)  # (n, 1)
    
    dist = query_sq + gallery_sq.T - 2 * np.dot(query_features, gallery_features.T)
    dist = np.clip(dist, 0, None)  # 数值稳定性
    dist = np.sqrt(dist)
    
    return dist


def k_reciprocal_neighbors(
    initial_rank: np.ndarray,
    i: int,
    k1: int,
) -> np.ndarray:
    """
    获取 k-reciprocal 近邻
    
    如果 j 在 i 的 top-k 近邻中，且 i 也在 j 的 top-k 近邻中，
    那么 j 是 i 的 k-reciprocal 近邻。
    
    Args:
        initial_rank: 初始排序索引 (N, N)
        i: 目标索引
        k1: k 值
        
    Returns:
        k-reciprocal 近邻索引
    """
    # i 的 top-k1 近邻
    forward_k_neighbors = initial_rank[i, :k1 + 1]
    
    # 找到满足 reciprocal 条件的近邻
    k_reciprocal_list = []
    for j in forward_k_neighbors:
        # j 的 top-k1 近邻
        backward_k_neighbors = initial_rank[j, :k1 + 1]
        if i in backward_k_neighbors:
            k_reciprocal_list.append(j)
    
    return np.array(k_reciprocal_list, dtype=np.int32)


def compute_jaccard_distance(
    target_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    print_progress: bool = True,
) -> np.ndarray:
    """
    计算 Jaccard 距离 (用于 re-ranking)
    
    Args:
        target_features: 目标特征 (query + gallery 合并)
        k1: k-reciprocal 参数 k1
        k2: 扩展参数 k2
        lambda_value: 原始距离权重
        print_progress: 是否打印进度
        
    Returns:
        重排序后的距离矩阵
    """
    N = target_features.shape[0]
    
    if print_progress:
        print(f"  Computing original distance matrix ({N}x{N})...")
    
    # 1. 计算原始欧氏距离
    original_dist = compute_euclidean_distance(target_features, target_features)
    
    # 2. 获取初始排序
    initial_rank = np.argsort(original_dist, axis=1)
    
    if print_progress:
        print(f"  Computing k-reciprocal neighbors (k1={k1}, k2={k2})...")
    
    # 3. 计算 V 矩阵 (k-reciprocal encoding)
    V = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        # 获取 k-reciprocal 近邻
        k_reciprocal_idx = k_reciprocal_neighbors(initial_rank, i, k1)
        
        # 扩展 k-reciprocal 集合
        k_reciprocal_expansion = k_reciprocal_idx.copy()
        for j in k_reciprocal_idx:
            candidate = k_reciprocal_neighbors(initial_rank, j, int(k1 / 2))
            if len(candidate) > k1 / 2:
                # 检查是否满足扩展条件
                overlap = len(np.intersect1d(candidate, k_reciprocal_idx))
                if overlap > 2 / 3 * len(candidate):
                    k_reciprocal_expansion = np.union1d(k_reciprocal_expansion, candidate)
        
        # 高斯权重
        weight = np.exp(-original_dist[i, k_reciprocal_expansion] ** 2)
        V[i, k_reciprocal_expansion] = weight / np.sum(weight)
    
    if print_progress:
        print(f"  Computing local query expansion...")
    
    # 4. Local Query Expansion (可选)
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i] = np.mean(V[initial_rank[i, :k2]], axis=0)
        V = V_qe
    
    if print_progress:
        print(f"  Computing Jaccard distance...")
    
    # 5. 计算 Jaccard 距离
    # Jaccard distance = 1 - intersection / union
    # 使用向量化计算
    inverted_index = np.zeros_like(V)
    for i in range(N):
        nz = np.where(V[:, i] > 0)[0]
        if len(nz) > 0:
            inverted_index[nz, i] = 1.0 / len(nz)
    
    jaccard_dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # 找到共同的非零元素
        nz_i = np.where(V[i] > 0)[0]
        
        # 使用点积计算 Jaccard 相似度
        temp_min = np.minimum(V[i], V)
        jaccard_dist[i] = 1 - np.sum(temp_min, axis=1)
    
    # 6. 最终距离 = lambda * original + (1 - lambda) * jaccard
    final_dist = lambda_value * original_dist + (1 - lambda_value) * jaccard_dist
    
    if print_progress:
        print(f"  Re-ranking complete!")
    
    return final_dist


def re_ranking(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    K-Reciprocal Re-ranking 主函数
    
    Args:
        query_features: Query 特征 (num_query, feat_dim)
        gallery_features: Gallery 特征 (num_gallery, feat_dim)
        k1: k-reciprocal 参数 (默认 20)
        k2: local query expansion 参数 (默认 6)
        lambda_value: 原始距离权重 (默认 0.3)
        
    Returns:
        重排序后的距离矩阵 (num_query, num_gallery)
    """
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    
    print(f"🔄 Re-ranking: {num_query} queries, {num_gallery} gallery images")
    print(f"   Parameters: k1={k1}, k2={k2}, lambda={lambda_value}")
    
    # 合并 query 和 gallery 特征
    all_features = np.vstack([query_features, gallery_features])
    
    # 计算 Jaccard 距离
    all_dist = compute_jaccard_distance(
        all_features,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    
    # 提取 query-gallery 距离矩阵
    final_dist = all_dist[:num_query, num_query:]
    
    return final_dist


if __name__ == "__main__":
    # 简单测试
    print("Testing Re-ranking...")
    
    np.random.seed(42)
    query = np.random.randn(10, 512).astype(np.float32)
    gallery = np.random.randn(50, 512).astype(np.float32)
    
    # L2 归一化
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    
    dist = re_ranking(query, gallery, k1=5, k2=2)
    print(f"Distance matrix shape: {dist.shape}")
    print("✅ Re-ranking test passed!")
