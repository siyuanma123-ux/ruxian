"""
使用真实数据集测试模型
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# 添加路径
sys.path.append(str(Path(__file__).parent))

from models.breast_cancer_model import BreastCancerDiagnosisModel
from data.cbis_breakhis_dataset import CBISBreakHisDataset


def test_model_with_real_data():
    """使用真实数据集测试模型"""
    
    print("=" * 60)
    print("使用真实数据集测试跨模态乳腺癌诊断模型")
    print("=" * 60)
    
    # 数据路径
    mammo_metadata = "x光数据集/manifest-1761561543948/metadata.csv"
    mammo_root = "x光数据集/manifest-1761561543948"
    patho_root = "病理数据集/BreaKHis_v1"
    
    # 检查路径
    if not Path(mammo_metadata).exists():
        print(f"错误: 找不到metadata文件: {mammo_metadata}")
        return
    
    if not Path(mammo_root).exists():
        print(f"错误: 找不到X光数据目录: {mammo_root}")
        return
    
    if not Path(patho_root).exists():
        print(f"错误: 找不到病理数据目录: {patho_root}")
        return
    
    # 创建数据集
    print("\n1. 加载数据集...")
    train_dataset = CBISBreakHisDataset(
        mammo_metadata=mammo_metadata,
        mammo_root=mammo_root,
        patho_root=patho_root,
        img_size=224,
        patch_size=16,
        use_pathology=True,
        max_samples=20,  # 限制样本数用于快速测试
        split='train'
    )
    
    test_dataset = CBISBreakHisDataset(
        mammo_metadata=mammo_metadata,
        mammo_root=mammo_root,
        patho_root=patho_root,
        img_size=224,
        patch_size=16,
        use_pathology=True,
        max_samples=10,
        split='test'
    )
    
    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 创建模型
    print("\n2. 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=6,  # 减少深度以加快测试
        num_heads=12,
        num_classes=2,  # 二分类：良性/恶性
        num_grades=2,
        use_dg=True,
        use_interpretability=True
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    print("\n3. 测试前向传播...")
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="测试前向传播")):
            mammo_img = batch['mammo_image'].to(device)
            patho_img = batch.get('patho_image')
            if patho_img is not None:
                patho_img = patho_img.to(device)
            
            # 前向传播
            outputs = model(mammo_img, patho_img, return_interpretability=True)
            
            print(f"\n批次 {batch_idx + 1}:")
            print(f"  输入: X光 {mammo_img.shape}, 病理 {patho_img.shape if patho_img is not None else 'None'}")
            print(f"  分类输出: {outputs['classification'].shape}")
            print(f"  定位输出: {outputs['localization'].shape}")
            print(f"  分级输出: {outputs['grading'].shape}")
            
            if 'heatmap' in outputs:
                print(f"  热图形状: {outputs['heatmap'].shape}")
            
            # 只测试第一个批次
            if batch_idx == 0:
                break
    
    # 测试损失计算
    print("\n4. 测试损失计算...")
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="测试损失计算")):
        mammo_img = batch['mammo_image'].to(device)
        patho_img = batch.get('patho_image')
        if patho_img is not None:
            patho_img = patho_img.to(device)
        
        label_cls = batch['label_cls'].to(device)
        label_grade = batch['label_grade'].to(device)
        bbox = batch['bbox'].to(device)
        
        # 前向传播
        outputs = model(mammo_img, patho_img)
        
        # 提取对齐特征
        mammo_proj = outputs.get('mammo_proj')
        patho_proj = outputs.get('patho_proj')
        
        # 计算损失
        total_loss, loss_dict = model.compute_loss(
            outputs,
            labels_cls=label_cls,
            labels_bbox=bbox,
            labels_grade=label_grade,
            labels_align=label_cls,
            mammo_features=mammo_proj,
            patho_features=patho_proj
        )
        
        print(f"\n批次 {batch_idx + 1} 损失:")
        print(f"  总损失: {total_loss.item():.4f}")
        for k, v in loss_dict.items():
            if k != 'task_weights' and k != 'total':
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.item():.4f}")
                else:
                    print(f"  {k}: {v}")
        
        # 只测试第一个批次
        if batch_idx == 0:
            break
    
    # 测试推理
    print("\n5. 测试推理...")
    model.eval()
    
    correct_cls = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试推理"):
            mammo_img = batch['mammo_image'].to(device)
            patho_img = batch.get('patho_image')
            if patho_img is not None:
                patho_img = patho_img.to(device)
            
            label_cls = batch['label_cls'].to(device)
            
            # 推理
            outputs = model(mammo_img, patho_img)
            
            # 计算准确率
            pred_cls = outputs['classification'].argmax(dim=1)
            correct_cls += (pred_cls == label_cls).sum().item()
            total_samples += label_cls.size(0)
    
    accuracy = correct_cls / total_samples if total_samples > 0 else 0.0
    print(f"\n测试准确率: {accuracy * 100:.2f}% ({correct_cls}/{total_samples})")
    
    # 测试可解释性
    print("\n6. 测试可解释性...")
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            mammo_img = batch['mammo_image'].to(device)
            patho_img = batch.get('patho_image')
            if patho_img is not None:
                patho_img = patho_img.to(device)
            
            # 获取可解释性结果
            outputs = model(mammo_img, patho_img, return_interpretability=True)
            
            if 'heatmap' in outputs:
                heatmap = outputs['heatmap'][0].cpu().numpy()
                print(f"热图形状: {heatmap.shape}")
                print(f"热图范围: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
            
            if 'top_patho_patches' in outputs:
                top_patches = outputs['top_patho_patches']
                print(f"Top-k病理patch: {top_patches.shape}")
            
            # 只测试第一个批次
            break
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    test_model_with_real_data()

