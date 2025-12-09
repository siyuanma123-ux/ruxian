"""
跨模态对齐模块
使用对比学习实现X光和病理特征的语义对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CrossModalAlignment(nn.Module):
    """跨模态对齐模块"""
    
    def __init__(self, embed_dim=768, temperature=0.07, projection_dim=256):
        super().__init__()
        self.temperature = temperature
        self.embed_dim = embed_dim
        
        # 投影层，将特征映射到对比学习空间
        self.mammo_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
        self.patho_proj = nn.Sequential(
            nn.Linear(embed_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(self, z_mammo, z_patho):
        """
        Args:
            z_mammo: [B, embed_dim] X光特征
            z_patho: [B, embed_dim] 病理特征
        Returns:
            mammo_proj: [B, projection_dim] 投影后的X光特征
            patho_proj: [B, projection_dim] 投影后的病理特征
        """
        mammo_proj = self.mammo_proj(z_mammo)
        patho_proj = self.patho_proj(z_patho)
        
        # L2归一化
        mammo_proj = F.normalize(mammo_proj, p=2, dim=1)
        patho_proj = F.normalize(patho_proj, p=2, dim=1)
        
        return mammo_proj, patho_proj


def compute_contrastive_loss(
    mammo_features,
    patho_features,
    labels=None,
    temperature=0.07
):
    """
    计算对比学习损失
    
    Args:
        mammo_features: [B, D] X光特征（已归一化）
        patho_features: [B, D] 病理特征（已归一化）
        labels: [B] 样本标签，用于构建正负样本对
        temperature: 温度参数
    Returns:
        loss: 对比损失标量
    """
    batch_size = mammo_features.shape[0]
    device = mammo_features.device
    
    # 计算相似度矩阵
    # mammo_features @ patho_features.T -> [B, B]
    logits = torch.matmul(mammo_features, patho_features.t()) / temperature
    
    # 构建正负样本对
    if labels is not None:
        # 同一标签的样本为正样本对
        labels = labels.unsqueeze(1)  # [B, 1]
        positive_mask = (labels == labels.t()).float()  # [B, B]
    else:
        # 默认：对角线为正样本对（同一病人的X光和病理）
        positive_mask = torch.eye(batch_size, device=device)
    
    # 正样本对标签
    labels_pos = torch.arange(batch_size, device=device)
    
    # InfoNCE损失（对称）
    loss_mammo = F.cross_entropy(logits, labels_pos)
    loss_patho = F.cross_entropy(logits.t(), labels_pos)
    
    loss = (loss_mammo + loss_patho) / 2.0
    
    return loss


def compute_alignment_accuracy(mammo_features, patho_features, labels=None, k=1):
    """
    计算对齐准确率（用于评估）
    
    Args:
        mammo_features: [B, D] X光特征
        patho_features: [B, D] 病理特征
        labels: [B] 样本标签
        k: Top-k准确率
    Returns:
        accuracy: 准确率
    """
    batch_size = mammo_features.shape[0]
    device = mammo_features.device
    
    # 计算相似度
    similarity = torch.matmul(mammo_features, patho_features.t())
    
    if labels is not None:
        # 基于标签的正样本对
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.t()).float()
    else:
        # 对角线为正样本对
        positive_mask = torch.eye(batch_size, device=device)
    
    # Top-k检索
    _, topk_indices = torch.topk(similarity, k, dim=1)
    
    correct = 0
    for i in range(batch_size):
        if labels is not None:
            # 检查top-k中是否有正样本
            topk_labels = labels[topk_indices[i]]
            if (topk_labels == labels[i]).any():
                correct += 1
        else:
            # 检查是否包含对角线元素
            if i in topk_indices[i]:
                correct += 1
    
    accuracy = correct / batch_size
    return accuracy

