"""
可解释性模块
通过跨模态注意力机制生成热图和病理patch关联
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, mammo_tokens, patho_tokens):
        """
        计算X光patch到病理patch的注意力
        
        Args:
            mammo_tokens: [B, N_m, embed_dim] X光patch tokens
            patho_tokens: [B, N_p, embed_dim] 病理patch tokens
        Returns:
            attended_mammo: [B, N_m, embed_dim] 注意力加权的X光特征
            attention_weights: [B, N_m, N_p] 注意力权重矩阵
        """
        B, N_m, _ = mammo_tokens.shape
        N_p = patho_tokens.shape[1]
        
        # Q来自X光，K和V来自病理
        Q = self.q_proj(mammo_tokens)  # [B, N_m, embed_dim]
        K = self.k_proj(patho_tokens)  # [B, N_p, embed_dim]
        V = self.v_proj(patho_tokens)  # [B, N_p, embed_dim]
        
        # 重塑为多头
        Q = Q.view(B, N_m, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_m, d]
        K = K.view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_p, d]
        V = V.view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N_p, d]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # [B, H, N_m, N_p]
        attention_weights = F.softmax(scores, dim=-1)  # [B, H, N_m, N_p]
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attended = torch.matmul(attention_weights, V)  # [B, H, N_m, d]
        
        # 合并多头
        attended = attended.transpose(1, 2).contiguous().view(B, N_m, self.embed_dim)  # [B, N_m, embed_dim]
        attended = self.out_proj(attended)
        
        # 平均所有头的注意力权重
        attention_weights_avg = attention_weights.mean(dim=1)  # [B, N_m, N_p]
        
        return attended, attention_weights_avg


class InterpretabilityModule(nn.Module):
    """可解释性模块（生成热图和病理patch关联）"""
    
    def __init__(self, embed_dim=768, num_heads=8, top_k=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.top_k = top_k
        
        self.attention = CrossModalAttention(embed_dim, num_heads)
        
    def compute_heatmap(self, attention_weights, patch_size=16, img_size=224):
        """
        从注意力权重生成热图
        
        Args:
            attention_weights: [B, N_m, N_p] 注意力权重
            patch_size: patch大小
            img_size: 图像大小
        Returns:
            heatmap: [B, H, W] 热图
        """
        B, N_m, N_p = attention_weights.shape
        
        # 对病理patch维度求平均，得到每个X光patch的重要性
        patch_importance = attention_weights.mean(dim=2)  # [B, N_m]
        
        # 重塑为空间维度
        num_patches_per_side = img_size // patch_size
        heatmap = patch_importance.view(B, num_patches_per_side, num_patches_per_side)
        
        # 上采样到原图大小
        heatmap = F.interpolate(
            heatmap.unsqueeze(1),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        # 归一化到[0, 1]
        heatmap = (heatmap - heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                  (heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - 
                   heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)
        
        return heatmap
    
    def get_top_pathology_patches(self, attention_weights, patho_tokens, top_k=None):
        """
        获取与X光最相关的top-k病理patch
        
        Args:
            attention_weights: [B, N_m, N_p] 注意力权重
            patho_tokens: [B, N_p, embed_dim] 病理patch tokens
            top_k: top-k数量
        Returns:
            top_patches: [B, top_k, embed_dim] top-k病理patch特征
            top_indices: [B, top_k] top-k索引
        """
        if top_k is None:
            top_k = self.top_k
        
        B, N_m, N_p = attention_weights.shape
        
        # 对X光patch维度求平均，得到每个病理patch的平均重要性
        patho_importance = attention_weights.mean(dim=1)  # [B, N_p]
        
        # 获取top-k
        top_values, top_indices = torch.topk(patho_importance, k=min(top_k, N_p), dim=1)  # [B, top_k]
        
        # 提取对应的patch特征
        B_idx = torch.arange(B, device=patho_tokens.device).unsqueeze(1).expand(-1, top_k)
        top_patches = patho_tokens[B_idx, top_indices]  # [B, top_k, embed_dim]
        
        return top_patches, top_indices
    
    def forward(self, mammo_tokens, patho_tokens, img_size=224, patch_size=16):
        """
        前向传播
        
        Args:
            mammo_tokens: [B, N_m, embed_dim] X光patch tokens
            patho_tokens: [B, N_p, embed_dim] 病理patch tokens
            img_size: 图像大小
            patch_size: patch大小
        Returns:
            heatmap: [B, H, W] 热图
            top_patho_patches: [B, top_k, embed_dim] top-k病理patch
            top_indices: [B, top_k] top-k索引
            attention_weights: [B, N_m, N_p] 注意力权重
        """
        # 计算跨模态注意力
        attended_mammo, attention_weights = self.attention(mammo_tokens, patho_tokens)
        
        # 生成热图
        heatmap = self.compute_heatmap(attention_weights, patch_size, img_size)
        
        # 获取top-k病理patch
        top_patho_patches, top_indices = self.get_top_pathology_patches(
            attention_weights, patho_tokens
        )
        
        return {
            'heatmap': heatmap,
            'top_patho_patches': top_patho_patches,
            'top_indices': top_indices,
            'attention_weights': attention_weights,
            'attended_mammo': attended_mammo
        }

