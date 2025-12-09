"""
多任务学习头
包含分类、定位、分级三个任务
使用不确定性加权法平衡多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class ClassificationHead(nn.Module):
    """分类头（四类：Normal, Benign, In situ, Invasive）"""
    
    def __init__(self, embed_dim=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, embed_dim] 特征
        Returns:
            logits: [B, num_classes] 分类logits
        """
        return self.head(features)


class LocalizationHead(nn.Module):
    """病灶定位头（边界框回归）"""
    
    def __init__(self, embed_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # [x, y, w, h]
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, embed_dim] 特征
        Returns:
            bbox: [B, 4] 边界框坐标（归一化到[0,1]）
        """
        bbox = self.head(features)
        bbox = torch.sigmoid(bbox)  # 归一化到[0,1]
        return bbox


class GradingHead(nn.Module):
    """病理分级头（序数回归）"""
    
    def __init__(self, embed_dim=768, num_grades=3, dropout=0.1):
        """
        Args:
            embed_dim: 特征维度
            num_grades: 分级数量（如：0=Benign, 1=In situ, 2=Invasive）
            dropout: Dropout率
        """
        super().__init__()
        self.num_grades = num_grades
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_grades)
        )
        
    def forward(self, features):
        """
        Args:
            features: [B, embed_dim] 特征
        Returns:
            logits: [B, num_grades] 分级logits
        """
        return self.head(features)


def compute_dice_loss(pred_mask, target_mask, smooth=1e-6):
    """
    计算Dice损失（用于定位任务）
    
    Args:
        pred_mask: [B, H, W] 预测mask
        target_mask: [B, H, W] 真实mask
        smooth: 平滑项
    Returns:
        dice_loss: Dice损失
    """
    pred_flat = pred_mask.view(pred_mask.shape[0], -1)
    target_flat = target_mask.view(target_mask.shape[0], -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()
    
    return dice_loss


def compute_bbox_loss(pred_bbox, target_bbox, reduction='mean'):
    """
    计算边界框损失（L1 + IoU）
    
    Args:
        pred_bbox: [B, 4] 预测边界框 [x, y, w, h]
        target_bbox: [B, 4] 真实边界框 [x, y, w, h]
        reduction: 损失归约方式
    Returns:
        loss: 边界框损失
    """
    # L1损失
    l1_loss = F.l1_loss(pred_bbox, target_bbox, reduction=reduction)
    
    # IoU损失
    def compute_iou(box1, box2):
        """计算IoU"""
        x1_min, y1_min = box1[:, 0], box1[:, 1]
        x1_max = x1_min + box1[:, 2]
        y1_max = y1_min + box1[:, 3]
        
        x2_min, y2_min = box2[:, 0], box2[:, 1]
        x2_max = x2_min + box2[:, 2]
        y2_max = y2_min + box2[:, 3]
        
        inter_x_min = torch.max(x1_min, x2_min)
        inter_y_min = torch.max(y1_min, y2_min)
        inter_x_max = torch.min(x1_max, x2_max)
        inter_y_max = torch.min(y1_max, y2_max)
        
        inter_area = torch.clamp(inter_x_max - inter_x_min, min=0) * \
                     torch.clamp(inter_y_max - inter_y_min, min=0)
        
        box1_area = box1[:, 2] * box1[:, 3]
        box2_area = box2[:, 2] * box2[:, 3]
        union_area = box1_area + box2_area - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    iou = compute_iou(pred_bbox, target_bbox)
    iou_loss = 1.0 - iou.mean()
    
    # 组合损失
    total_loss = l1_loss + iou_loss
    
    return total_loss


def compute_ordinal_loss(pred_logits, target_grades):
    """
    计算序数回归损失（相邻等级惩罚轻，远距离惩罚重）
    
    Args:
        pred_logits: [B, num_grades] 预测logits
        target_grades: [B] 真实等级（整数）
    Returns:
        loss: 序数损失
    """
    num_grades = pred_logits.shape[1]
    probs = F.softmax(pred_logits, dim=1)
    
    # 构建序数标签（one-hot）
    target_onehot = F.one_hot(target_grades, num_classes=num_grades).float()
    
    # 计算每个等级的距离权重
    # 距离越远，权重越大
    weights = torch.zeros(num_grades, num_grades, device=pred_logits.device)
    for i in range(num_grades):
        for j in range(num_grades):
            weights[i, j] = abs(i - j) + 1  # 距离+1，避免0权重
    
    # 加权交叉熵
    loss = 0.0
    for i in range(num_grades):
        weight = weights[target_grades, i]
        loss += weight * F.cross_entropy(
            pred_logits, 
            torch.full((pred_logits.shape[0],), i, device=pred_logits.device),
            reduction='none'
        )
    
    loss = loss.mean()
    return loss


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    不确定性加权的多任务损失
    每个任务有一个可学习的权重参数σ
    """
    
    def __init__(self, num_tasks=3):
        super().__init__()
        # 学习任务的不确定性（log方差）
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        """
        Args:
            losses: List of losses for each task
        Returns:
            weighted_loss: 加权后的总损失
            task_weights: 各任务的权重
        """
        weighted_losses = []
        task_weights = []
        
        for i, loss in enumerate(losses):
            # 权重 = 1 / (2 * σ^2)
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
            task_weights.append(precision.item())
        
        total_loss = sum(weighted_losses)
        
        return total_loss, task_weights

