"""
使用示例：跨模态乳腺癌诊断模型
"""

import torch
import torch.nn as nn
from models import BreastCancerDiagnosisModel
from models.causal_tta import CausalTTA


def example_basic_usage():
    """基本使用示例"""
    print("=" * 50)
    print("基本使用示例")
    print("=" * 50)
    
    # 创建模型
    model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=4,
        num_grades=3,
        use_dg=True,
        use_interpretability=True
    )
    
    # 创建模拟输入
    batch_size = 2
    mammo_img = torch.randn(batch_size, 1, 224, 224)  # X光图像（单通道）
    patho_img = torch.randn(batch_size, 3, 224, 224)  # 病理图像（RGB）
    
    # 前向传播
    outputs = model(mammo_img, patho_img, return_interpretability=True)
    
    print(f"输入形状: X光 {mammo_img.shape}, 病理 {patho_img.shape}")
    print(f"分类输出: {outputs['classification'].shape}")
    print(f"定位输出: {outputs['localization'].shape}")
    print(f"分级输出: {outputs['grading'].shape}")
    
    if 'heatmap' in outputs:
        print(f"热图形状: {outputs['heatmap'].shape}")
    
    # 计算损失
    labels_cls = torch.tensor([0, 1])
    labels_grade = torch.tensor([0, 1])
    labels_bbox = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]])
    
    mammo_proj = outputs.get('mammo_proj')
    patho_proj = outputs.get('patho_proj')
    
    total_loss, loss_dict = model.compute_loss(
        outputs,
        labels_cls=labels_cls,
        labels_bbox=labels_bbox,
        labels_grade=labels_grade,
        labels_align=labels_cls,
        mammo_features=mammo_proj,
        patho_features=patho_proj
    )
    
    print(f"\n总损失: {total_loss.item():.4f}")
    print("各损失项:")
    for k, v in loss_dict.items():
        if k != 'task_weights':
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.item():.4f}")
            else:
                print(f"  {k}: {v}")
    
    print("\n✓ 基本使用示例完成")


def example_tta_usage():
    """测试时自适应使用示例"""
    print("\n" + "=" * 50)
    print("测试时自适应使用示例")
    print("=" * 50)
    
    # 创建模型
    model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768
    )
    
    # 创建TTA适配器
    tta = CausalTTA(model, lr=1e-4, entropy_weight=0.1)
    
    # 模拟目标域数据（无标签）
    target_data = torch.randn(4, 1, 224, 224)
    
    # 执行自适应（这里只演示一步）
    loss = tta.adapt_step(target_data)
    print(f"TTA损失: {loss:.4f}")
    
    print("✓ 测试时自适应示例完成")


def example_interpretability():
    """可解释性使用示例"""
    print("\n" + "=" * 50)
    print("可解释性使用示例")
    print("=" * 50)
    
    model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        use_interpretability=True
    )
    
    mammo_img = torch.randn(1, 1, 224, 224)
    patho_img = torch.randn(1, 3, 224, 224)
    
    # 获取可解释性结果
    outputs = model(mammo_img, patho_img, return_interpretability=True)
    
    heatmap = outputs['heatmap']
    top_patches = outputs['top_patho_patches']
    top_indices = outputs['top_indices']
    
    print(f"热图形状: {heatmap.shape}")
    print(f"Top-k病理patch形状: {top_patches.shape}")
    print(f"Top-k索引: {top_indices}")
    print(f"注意力权重形状: {outputs['attention_weights'].shape}")
    
    print("✓ 可解释性示例完成")


if __name__ == '__main__':
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行示例
    example_basic_usage()
    example_tta_usage()
    example_interpretability()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")
    print("=" * 50)

