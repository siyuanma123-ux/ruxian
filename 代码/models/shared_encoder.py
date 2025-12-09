"""
共享视觉Transformer编码器模块
支持X光和病理两种模态的统一特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PatchEmbedding(nn.Module):
    """图像patch嵌入层"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AdapterLayer(nn.Module):
    """轻量级Adapter模块，用于保留模态特定特征"""
    
    def __init__(self, dim, adapter_dim=None, dropout=0.1):
        super().__init__()
        if adapter_dim is None:
            adapter_dim = dim // 4
        
        self.down_proj = nn.Linear(dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x


class ChannelMLP(nn.Module):
    """通道MLP（ViT中的Feed-Forward Network）"""
    
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PermuteMLP(nn.Module):
    """Permute-MLP（用于增强空间感知）"""
    
    def __init__(self, dim, mlp_ratio=0.5, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码块（带Adapter）"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, adapter_dim=None, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.adapter1 = AdapterLayer(dim, adapter_dim, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = ChannelMLP(dim, mlp_ratio, dropout)
        self.adapter2 = AdapterLayer(dim, adapter_dim, dropout)
        
        self.norm3 = nn.LayerNorm(dim)
        self.permute_mlp = PermuteMLP(dim, 0.5, dropout)
        self.norm4 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-attention with adapter
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x)
        x = residual + x_attn
        x = self.adapter1(x)
        
        # Channel MLP
        residual = x
        x = self.norm2(x)
        x = self.channel_mlp(x)
        x = residual + x
        x = self.adapter2(x)
        
        # Permute MLP
        residual = x
        x = self.norm3(x)
        x = self.permute_mlp(x)
        x = residual + x
        x = self.norm4(x)
        
        return x


class SharedViTEncoder(nn.Module):
    """共享视觉Transformer编码器"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        adapter_dim=None,
        dropout=0.1,
        modality_type='mammography'
    ):
        super().__init__()
        self.modality_type = modality_type
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Modality-specific embedding (用于区分X光和病理)
        self.modality_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Dropout
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, adapter_dim, dropout)
            for _ in range(depth)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        
    def forward(self, x, return_patch_tokens=False):
        """
        Args:
            x: [B, C, H, W] 输入图像
            return_patch_tokens: 是否返回patch tokens（用于可解释性）
        Returns:
            cls_token: [B, embed_dim] CLS token特征
            patch_tokens: [B, N, embed_dim] patch tokens（可选）
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Add modality embedding
        modality_tokens = self.modality_embed.expand(B, -1, -1)
        x = torch.cat([modality_tokens, x], dim=1)
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract CLS token (first token after modality token)
        cls_token = x[:, 1, :]  # [B, embed_dim]
        
        if return_patch_tokens:
            patch_tokens = x[:, 2:, :]  # [B, N, embed_dim]
            return cls_token, patch_tokens
        else:
            return cls_token

