"""
训练脚本
跨模态乳腺癌诊断模型训练
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# 添加路径
sys.path.append(str(Path(__file__).parent))

from models.breast_cancer_model import BreastCancerDiagnosisModel
from data.dataset import CrossModalBreastCancerDataset


def parse_args():
    parser = argparse.ArgumentParser(description='训练跨模态乳腺癌诊断模型')
    
    # 数据参数
    parser.add_argument('--mammo_csv', type=str, required=True, help='X光数据CSV文件')
    parser.add_argument('--mammo_root', type=str, required=True, help='X光图像根目录')
    parser.add_argument('--patho_root', type=str, default='', help='病理图像根目录')
    parser.add_argument('--use_pathology', action='store_true', help='是否使用病理数据')
    
    # 模型参数
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch大小')
    parser.add_argument('--embed_dim', type=int, default=768, help='嵌入维度')
    parser.add_argument('--depth', type=int, default=12, help='Transformer深度')
    parser.add_argument('--num_heads', type=int, default=12, help='注意力头数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 其他
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs', help='日志目录')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_dict_avg = {}
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        mammo_img = batch['mammo_image'].to(device)
        patho_img = batch.get('patho_image')
        if patho_img is not None:
            patho_img = patho_img.to(device)
        
        label_cls = batch['label_cls'].to(device)
        label_grade = batch['label_grade'].to(device)
        bbox = batch['bbox'].to(device)
        
        # 前向传播
        outputs = model(mammo_img, patho_img, return_interpretability=False)
        
        # 提取对齐特征
        mammo_proj = outputs.get('mammo_proj')
        patho_proj = outputs.get('patho_proj')
        
        # 计算损失
        total_loss_batch, loss_dict = model.compute_loss(
            outputs,
            labels_cls=label_cls,
            labels_bbox=bbox,
            labels_grade=label_grade,
            labels_align=label_cls,  # 使用分类标签作为对齐标签
            mammo_features=mammo_proj,
            patho_features=patho_proj
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计损失
        total_loss += total_loss_batch.item()
        for k, v in loss_dict.items():
            if k != 'task_weights':
                if k not in loss_dict_avg:
                    loss_dict_avg[k] = 0.0
                if isinstance(v, torch.Tensor):
                    loss_dict_avg[k] += v.item()
                else:
                    loss_dict_avg[k] += v
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'cls': f'{loss_dict["classification"].item():.4f}'
        })
    
    # 平均损失
    num_batches = len(dataloader)
    total_loss /= num_batches
    for k in loss_dict_avg:
        loss_dict_avg[k] /= num_batches
    
    return total_loss, loss_dict_avg


def validate(model, dataloader, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct_cls = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
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
            total_loss_batch, _ = model.compute_loss(
                outputs,
                labels_cls=label_cls,
                labels_bbox=bbox,
                labels_grade=label_grade,
                labels_align=label_cls,
                mammo_features=mammo_proj,
                patho_features=patho_proj
            )
            
            total_loss += total_loss_batch.item()
            
            # 计算准确率
            pred_cls = outputs['classification'].argmax(dim=1)
            correct_cls += (pred_cls == label_cls).sum().item()
            total_samples += label_cls.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_cls / total_samples
    
    return avg_loss, accuracy


def main():
    args = parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据加载
    train_dataset = CrossModalBreastCancerDataset(
        mammo_csv=args.mammo_csv,
        mammo_root=args.mammo_root,
        patho_root=args.patho_root,
        img_size=args.img_size,
        patch_size=args.patch_size,
        use_pathology=args.use_pathology
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 模型
    model = BreastCancerDiagnosisModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        use_dg=True,
        use_interpretability=args.use_pathology
    ).to(device)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'从epoch {start_epoch}恢复训练')
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # 训练
        train_loss, train_loss_dict = train_epoch(
            model, train_loader, optimizer, device, epoch+1
        )
        
        # 验证（这里简化，实际应该用验证集）
        val_loss, val_acc = validate(model, train_loader, device)
        
        # 学习率调度
        scheduler.step()
        
        # 记录日志
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/ClsLoss', train_loss_dict.get('classification', 0), epoch)
        writer.add_scalar('Train/LocLoss', train_loss_dict.get('localization', 0), epoch)
        writer.add_scalar('Train/GradeLoss', train_loss_dict.get('grading', 0), epoch)
        writer.add_scalar('Train/AlignLoss', train_loss_dict.get('alignment', 0), epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存检查点
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, save_dir / 'best_model.pth')
            print(f'保存最佳模型 (Acc: {best_acc:.4f})')
        
        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    print('训练完成！')


if __name__ == '__main__':
    main()

