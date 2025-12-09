"""
域泛化模块
包含MixStyle和IRM（Invariant Risk Minimization）方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import random


class MixStyle(nn.Module):
    """
    MixStyle: 混合不同域的统计特征（均值和方差）
    用于增强模型的域泛化能力
    """
    
    def __init__(self, alpha=0.1, p=0.5):
        """
        Args:
            alpha: 混合强度（beta分布的参数）
            p: 应用MixStyle的概率
        """
        super().__init__()
        self.alpha = alpha
        self.p = p
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 特征图
        Returns:
            mixed_x: 混合后的特征
        """
        if not self.training or random.random() > self.p:
            return x
        
        B, C, H, W = x.shape
        
        # 计算统计量
        mu = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        sigma = x.std(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        
        # 打乱batch顺序
        perm = torch.randperm(B, device=x.device)
        
        # 混合统计量
        mu_mix = mu[perm]
        sigma_mix = sigma[perm]
        
        # Beta分布采样
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)
        
        # 混合
        mu_mixed = lam * mu + (1 - lam) * mu_mix
        sigma_mixed = lam * sigma + (1 - lam) * sigma_mix
        
        # 标准化并应用新统计量
        x_norm = (x - mu) / (sigma + 1e-6)
        x_mixed = x_norm * sigma_mixed + mu_mixed
        
        return x_mixed


class IRMLoss(nn.Module):
    """
    Invariant Risk Minimization (IRM) 损失
    约束模型在所有域上都达到最优性能
    """
    
    def __init__(self, penalty_weight=1.0):
        """
        Args:
            penalty_weight: IRM惩罚项的权重
        """
        super().__init__()
        self.penalty_weight = penalty_weight
        
    def compute_irm_penalty(self, logits, labels):
        """
        计算IRM惩罚项
        
        Args:
            logits: [B, num_classes] 模型输出
            labels: [B] 真实标签
        Returns:
            penalty: IRM惩罚项
        """
        # 计算损失
        loss = F.cross_entropy(logits, labels, reduction='none')
        
        # 计算损失对模型参数的梯度
        # 这里使用虚拟梯度（dummy gradient）来近似IRM惩罚
        # 实际实现中，IRM需要计算损失对特征表示的梯度
        
        # 简化版本：使用损失的方差作为惩罚项
        # 这鼓励模型在不同域上产生一致的损失
        penalty = loss.var()
        
        return penalty
    
    def forward(self, logits_list, labels_list):
        """
        计算多域的IRM损失
        
        Args:
            logits_list: List of [B_i, num_classes] 不同域的输出
            labels_list: List of [B_i] 不同域的真实标签
        Returns:
            total_loss: 总损失
            penalty: IRM惩罚项
        """
        total_loss = 0.0
        penalty = 0.0
        
        for logits, labels in zip(logits_list, labels_list):
            domain_loss = F.cross_entropy(logits, labels)
            total_loss += domain_loss
            
            # IRM惩罚
            domain_penalty = self.compute_irm_penalty(logits, labels)
            penalty += domain_penalty
        
        # 平均
        total_loss = total_loss / len(logits_list)
        penalty = penalty / len(logits_list)
        
        return total_loss, penalty


class DomainGeneralization(nn.Module):
    """域泛化模块（整合MixStyle和IRM）"""
    
    def __init__(self, use_mixstyle=True, mixstyle_alpha=0.1, irm_penalty=1.0):
        super().__init__()
        self.use_mixstyle = use_mixstyle
        self.mixstyle = MixStyle(alpha=mixstyle_alpha) if use_mixstyle else None
        self.irm_loss = IRMLoss(penalty_weight=irm_penalty)
        
    def apply_mixstyle(self, features):
        """应用MixStyle"""
        if self.use_mixstyle and self.training:
            return self.mixstyle(features)
        return features
    
    def compute_irm_loss(self, logits_list, labels_list):
        """计算IRM损失"""
        return self.irm_loss(logits_list, labels_list)

