"""
跨模态乳腺癌诊断模型
整合所有模块：共享编码器、跨模态对齐、域泛化、多任务学习、可解释性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .shared_encoder import SharedViTEncoder
from .cross_modal_alignment import CrossModalAlignment, compute_contrastive_loss
from .domain_generalization import DomainGeneralization
from .multi_task_heads import (
    ClassificationHead, LocalizationHead, GradingHead,
    UncertaintyWeightedMultiTaskLoss, compute_bbox_loss, compute_ordinal_loss
)
from .interpretability import InterpretabilityModule


class BreastCancerDiagnosisModel(nn.Module):
    """跨模态乳腺癌诊断模型"""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=4,
        num_grades=3,
        use_dg=True,
        use_interpretability=True,
        projection_dim=256,
        temperature=0.07
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_dg = use_dg
        self.use_interpretability = use_interpretability
        
        # 共享编码器（X光和病理共用）
        self.mammo_encoder = SharedViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=1,  # X光是灰度图
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            modality_type='mammography'
        )
        
        self.patho_encoder = SharedViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,  # 病理是RGB
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            modality_type='pathology'
        )
        
        # 跨模态对齐模块
        self.alignment = CrossModalAlignment(
            embed_dim=embed_dim,
            temperature=temperature,
            projection_dim=projection_dim
        )
        
        # 域泛化模块
        if use_dg:
            self.domain_gen = DomainGeneralization(
                use_mixstyle=True,
                mixstyle_alpha=0.1,
                irm_penalty=1.0
            )
        else:
            self.domain_gen = None
        
        # 多任务学习头
        self.cls_head = ClassificationHead(embed_dim, num_classes)
        self.loc_head = LocalizationHead(embed_dim)
        self.grade_head = GradingHead(embed_dim, num_grades)
        
        # 不确定性加权损失
        self.uncertainty_loss = UncertaintyWeightedMultiTaskLoss(num_tasks=3)
        
        # 可解释性模块
        if use_interpretability:
            self.interpretability = InterpretabilityModule(
                embed_dim=embed_dim,
                num_heads=num_heads,
                top_k=5
            )
        else:
            self.interpretability = None
        
    def forward(
        self,
        mammo_img,
        patho_img=None,
        return_interpretability=False,
        return_patch_tokens=False
    ):
        """
        前向传播
        
        Args:
            mammo_img: [B, 1, H, W] X光图像
            patho_img: [B, 3, H, W] 病理图像（可选）
            return_interpretability: 是否返回可解释性结果
            return_patch_tokens: 是否返回patch tokens
        Returns:
            outputs: 包含所有输出的字典
        """
        outputs = {}
        
        # X光编码
        if return_patch_tokens or return_interpretability:
            z_mammo, mammo_tokens = self.mammo_encoder(
                mammo_img, return_patch_tokens=True
            )
            outputs['mammo_tokens'] = mammo_tokens
        else:
            z_mammo = self.mammo_encoder(mammo_img)
        
        outputs['z_mammo'] = z_mammo
        
        # 病理编码（如果有）
        if patho_img is not None:
            if return_patch_tokens or return_interpretability:
                z_patho, patho_tokens = self.patho_encoder(
                    patho_img, return_patch_tokens=True
                )
                outputs['patho_tokens'] = patho_tokens
            else:
                z_patho = self.patho_encoder(patho_img)
            
            outputs['z_patho'] = z_patho
            
            # 跨模态对齐
            mammo_proj, patho_proj = self.alignment(z_mammo, z_patho)
            outputs['mammo_proj'] = mammo_proj
            outputs['patho_proj'] = patho_proj
            
            # 可解释性（如果启用）
            if return_interpretability and self.interpretability is not None:
                interp_results = self.interpretability(
                    mammo_tokens, patho_tokens, self.img_size, self.patch_size
                )
                outputs.update(interp_results)
        
        # 多任务预测（使用X光特征，或融合特征）
        if patho_img is not None:
            # 融合特征（简单拼接后投影）
            fused_features = z_mammo + z_patho  # 或使用更复杂的融合方式
        else:
            fused_features = z_mammo
        
        # 分类
        cls_logits = self.cls_head(fused_features)
        outputs['classification'] = cls_logits
        
        # 定位
        bbox_pred = self.loc_head(fused_features)
        outputs['localization'] = bbox_pred
        
        # 分级
        grade_logits = self.grade_head(fused_features)
        outputs['grading'] = grade_logits
        
        return outputs
    
    def compute_loss(
        self,
        outputs,
        labels_cls=None,
        labels_bbox=None,
        labels_grade=None,
        labels_align=None,
        mammo_features=None,
        patho_features=None
    ):
        """
        计算总损失
        
        Args:
            outputs: 模型输出字典
            labels_cls: [B] 分类标签
            labels_bbox: [B, 4] 边界框标签
            labels_grade: [B] 分级标签
            labels_align: [B] 对齐标签（用于对比学习）
            mammo_features: [B, D] X光特征（用于对齐）
            patho_features: [B, D] 病理特征（用于对齐）
        Returns:
            total_loss: 总损失
            loss_dict: 各损失项的字典
        """
        loss_dict = {}
        
        # 分类损失
        if labels_cls is not None:
            loss_cls = F.cross_entropy(outputs['classification'], labels_cls)
            loss_dict['classification'] = loss_cls
        else:
            loss_cls = torch.tensor(0.0, device=outputs['classification'].device)
            loss_dict['classification'] = loss_cls
        
        # 定位损失
        if labels_bbox is not None:
            loss_loc = compute_bbox_loss(outputs['localization'], labels_bbox)
            loss_dict['localization'] = loss_loc
        else:
            loss_loc = torch.tensor(0.0, device=outputs['localization'].device)
            loss_dict['localization'] = loss_loc
        
        # 分级损失
        if labels_grade is not None:
            loss_grade = compute_ordinal_loss(outputs['grading'], labels_grade)
            loss_dict['grading'] = loss_grade
        else:
            loss_grade = torch.tensor(0.0, device=outputs['grading'].device)
            loss_dict['grading'] = loss_grade
        
        # 跨模态对齐损失
        if mammo_features is not None and patho_features is not None:
            loss_align = compute_contrastive_loss(
                mammo_features,
                patho_features,
                labels=labels_align,
                temperature=0.07
            )
            loss_dict['alignment'] = loss_align
        else:
            loss_align = torch.tensor(0.0, device=outputs['classification'].device)
            loss_dict['alignment'] = loss_align
        
        # 不确定性加权多任务损失
        task_losses = [loss_cls, loss_loc, loss_grade]
        total_loss, task_weights = self.uncertainty_loss(task_losses)
        
        # 加上对齐损失
        total_loss = total_loss + 0.1 * loss_align  # 对齐损失权重
        
        loss_dict['total'] = total_loss
        loss_dict['task_weights'] = task_weights
        
        return total_loss, loss_dict

