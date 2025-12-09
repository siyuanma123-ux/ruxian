"""
因果测试时自适应（Causal Test-Time Adaptation）
只在推理时更新BN层统计参数和Adapter权重，保持诊断路径不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import copy


class CausalTTA:
    """
    因果测试时自适应
    只更新特征统计部分（BN层、Adapter），不更新诊断路径
    """
    
    def __init__(self, model, lr=1e-4, entropy_weight=0.1):
        """
        Args:
            model: 要自适应的模型
            lr: 学习率
            entropy_weight: 熵最小化权重
        """
        self.model = model
        self.lr = lr
        self.entropy_weight = entropy_weight
        
        # 只优化BN层和Adapter层
        self.optimizable_params = []
        seen_params = set()
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                # BN层参数
                if hasattr(module, 'weight') and module.weight is not None:
                    param_id = id(module.weight)
                    if param_id not in seen_params:
                        self.optimizable_params.append(module.weight)
                        seen_params.add(param_id)
                if hasattr(module, 'bias') and module.bias is not None:
                    param_id = id(module.bias)
                    if param_id not in seen_params:
                        self.optimizable_params.append(module.bias)
                        seen_params.add(param_id)
            elif 'adapter' in name.lower():
                # Adapter层参数
                for param in module.parameters():
                    if param.requires_grad:
                        param_id = id(param)
                        if param_id not in seen_params:
                            self.optimizable_params.append(param)
                            seen_params.add(param_id)
        
        # 创建优化器（只优化这些参数）
        self.optimizer = torch.optim.Adam(self.optimizable_params, lr=lr)
        
        # 保存源域统计量（用于约束）
        self.source_stats = {}
        self._save_source_stats()
        
    def _save_source_stats(self):
        """保存源域的BN统计量"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                    self.source_stats[name] = {
                        'mean': module.running_mean.clone(),
                        'var': module.running_var.clone()
                    }
    
    def compute_entropy_loss(self, logits):
        """
        计算预测熵（用于最小化，使预测更确定）
        
        Args:
            logits: [B, num_classes] 模型输出
        Returns:
            entropy_loss: 熵损失
        """
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        return entropy
    
    def compute_source_constraint(self):
        """
        计算源域约束（防止统计量偏离太远）
        
        Returns:
            constraint_loss: 约束损失
        """
        constraint_loss = 0.0
        count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in self.source_stats:
                    source_mean = self.source_stats[name]['mean']
                    source_var = self.source_stats[name]['var']
                    
                    current_mean = module.running_mean
                    current_var = module.running_var
                    
                    # L2距离
                    mean_diff = F.mse_loss(current_mean, source_mean)
                    var_diff = F.mse_loss(current_var, source_var)
                    
                    constraint_loss += mean_diff + var_diff
                    count += 1
        
        if count > 0:
            constraint_loss = constraint_loss / count
        
        return constraint_loss
    
    def adapt_step(self, x, source_constraint_weight=0.01):
        """
        执行一步自适应
        
        Args:
            x: [B, C, H, W] 输入图像（无标签）
            source_constraint_weight: 源域约束权重
        Returns:
            loss: 总损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        outputs = self.model(x)
        
        # 如果是字典，提取logits
        if isinstance(outputs, dict):
            logits = outputs.get('classification', outputs.get('logits'))
        else:
            logits = outputs
        
        # 熵最小化损失
        entropy_loss = self.compute_entropy_loss(logits)
        
        # 源域约束
        constraint_loss = self.compute_source_constraint()
        
        # 总损失
        total_loss = self.entropy_weight * entropy_loss + source_constraint_weight * constraint_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def adapt(self, dataloader, num_steps=100):
        """
        在目标域数据上自适应
        
        Args:
            dataloader: 目标域数据加载器（无标签）
            num_steps: 自适应步数
        """
        self.model.train()
        
        step = 0
        for batch in dataloader:
            if step >= num_steps:
                break
            
            # 提取图像（假设batch是字典或元组）
            if isinstance(batch, dict):
                x = batch['image']
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            
            x = x.cuda() if next(self.model.parameters()).is_cuda else x
            
            # 自适应步骤
            loss = self.adapt_step(x)
            
            step += 1
            
            if step % 10 == 0:
                print(f"TTA Step {step}/{num_steps}, Loss: {loss:.4f}")
    
    def reset(self):
        """重置到初始状态"""
        # 恢复源域统计量
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if name in self.source_stats:
                    module.running_mean.data = self.source_stats[name]['mean'].clone()
                    module.running_var.data = self.source_stats[name]['var'].clone()

