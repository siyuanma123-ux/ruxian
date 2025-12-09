[README.md](https://github.com/user-attachments/files/24051356/README.md)
# Models 模块使用文档

这个目录包含了跨模态乳腺癌诊断系统的所有核心模型代码。每个文件都是独立的模块，可以单独使用，也可以组合起来构建完整的诊断系统。

## 目录结构

```
models/
├── __init__.py                    # 统一导出接口
├── shared_encoder.py              # 共享ViT编码器，处理X光和病理图像
├── cross_modal_alignment.py      # 跨模态对齐，让两种模态特征对齐
├── domain_generalization.py      # 域泛化，提高模型鲁棒性
├── causal_tta.py                 # 测试时自适应，适应新数据
├── multi_task_heads.py           # 多任务学习头，分类/定位/分级
├── interpretability.py          # 可解释性，生成热图和注意力
└── breast_cancer_model.py        # 主模型，整合所有模块
```

---

## 1. shared_encoder.py - 共享编码器

这个文件实现了统一的视觉Transformer编码器，用来处理X光图像和病理图像。核心思想是让两种模态共享同一个编码器，但通过Adapter机制保留各自的模态特性。

### SharedViTEncoder

这是主要的编码器类。它基于Vision Transformer架构，但做了一些修改来适配我们的任务。

**初始化参数：**

- `img_size`: 输入图像大小，默认224。注意这里假设输入是正方形，如果不是需要先resize。
- `patch_size`: 图像会被切分成patch，这个参数控制patch大小，默认16。224x224的图像会被切成14x14=196个patch。
- `in_channels`: 输入通道数。X光图像是1通道（灰度），病理图像是3通道（RGB）。这个参数必须正确设置，否则会报错。
- `embed_dim`: 特征嵌入维度，默认768。这个值决定了模型容量，越大模型越强但越慢。
- `depth`: Transformer层数，默认12。层数越多模型越深，但训练越慢。
- `num_heads`: 多头注意力的头数，默认12。通常embed_dim要能被num_heads整除。
- `mlp_ratio`: MLP扩展比例，默认4.0。这是ViT的标准设置。
- `adapter_dim`: Adapter的维度，默认None会自动设为embed_dim//4。Adapter是用来保留模态特性的，维度太小可能不够用。
- `dropout`: Dropout率，默认0.1。训练时随机丢弃一些神经元防止过拟合。
- `modality_type`: 模态类型字符串，'mammography'或'pathology'。这个参数目前主要用于区分，实际影响不大。

**使用示例：**

```python
from models.shared_encoder import SharedViTEncoder

# 创建X光编码器
mammo_encoder = SharedViTEncoder(
    img_size=224,
    patch_size=16,
    in_channels=1,  # 单通道
    embed_dim=768,
    depth=12,
    num_heads=12,
    modality_type='mammography'
)

# 创建病理编码器（注意in_channels=3）
patho_encoder = SharedViTEncoder(
    img_size=224,
    patch_size=16,
    in_channels=3,  # RGB三通道
    embed_dim=768,
    depth=12,
    num_heads=12,
    modality_type='pathology'
)

# 前向传播
mammo_img = torch.randn(2, 1, 224, 224)  # batch_size=2
z_mammo = mammo_encoder(mammo_img)  # 输出 [2, 768]

# 如果需要patch tokens（用于可解释性）
z_mammo, mammo_tokens = mammo_encoder(mammo_img, return_patch_tokens=True)
# z_mammo: [2, 768] - CLS token
# mammo_tokens: [2, 196, 768] - 所有patch的token
```

**注意事项：**

1. 输入图像必须是torch.Tensor格式，且已经归一化。通常X光图像归一化到[-1, 1]，病理图像用ImageNet的均值和方差。
2. 如果图像不是224x224，需要先resize。可以用torchvision.transforms.Resize。
3. `return_patch_tokens=True`会增加内存消耗，因为要保存所有patch的特征。如果只是做分类，不需要这个。
4. 编码器的输出是CLS token，这是整个图像的全局特征表示。

### AdapterLayer

这是Adapter模块，用来在共享编码器中保留模态特定特征。设计思路是：共享编码器学习通用特征，Adapter学习模态特定的校准。

**工作原理：**

Adapter采用瓶颈结构：输入 → 降维 → 激活 → 升维 → 输出，然后和输入做残差连接。这样既能学习新特征，又不会破坏原有特征。

```python
from models.shared_encoder import AdapterLayer

# 在TransformerBlock内部使用，不需要单独创建
# 但如果你想了解它的结构：
adapter = AdapterLayer(dim=768, adapter_dim=192, dropout=0.1)
x = torch.randn(10, 196, 768)  # [batch, num_patches, embed_dim]
x_out = adapter(x)  # 输出形状和输入相同
```

**为什么需要Adapter：**

如果X光和病理完全共享编码器，两种模态的特征可能会过度混合，失去各自的特性。比如X光的钙化点和病理的细胞形态是完全不同的特征，如果混在一起可能学不好。Adapter让模型既能共享通用特征（比如边缘、纹理），又能保留模态特定特征。

### TransformerBlock

这是Transformer的编码块，包含自注意力、MLP、Adapter等组件。通常不需要直接使用，但了解结构有助于调试。

**结构：**
1. LayerNorm + Self-Attention + Adapter
2. LayerNorm + ChannelMLP + Adapter  
3. LayerNorm + PermuteMLP

每个组件都有残差连接，这是Transformer的标准设计。

---

## 2. cross_modal_alignment.py - 跨模态对齐

这个模块负责让X光和病理的特征在语义空间中对齐。核心思想是：同一病人的X光和病理图像应该映射到相近的特征空间位置。

### CrossModalAlignment

这个类将X光和病理特征投影到对比学习空间，然后通过对比学习让它们对齐。

**初始化参数：**

- `embed_dim`: 输入特征维度，默认768。必须和编码器输出的维度一致。
- `temperature`: 温度参数，默认0.07。这个参数控制对比学习的"软硬"程度，越小越hard，越大越soft。0.07是常用的值。
- `projection_dim`: 投影后的维度，默认256。通常比embed_dim小，这样可以降维并学习更紧凑的表示。

**使用示例：**

```python
from models.cross_modal_alignment import CrossModalAlignment, compute_contrastive_loss

# 创建对齐模块
alignment = CrossModalAlignment(embed_dim=768, projection_dim=256, temperature=0.07)

# 假设已经有了编码后的特征
z_mammo = torch.randn(4, 768)  # 4个X光样本的特征
z_patho = torch.randn(4, 768)  # 4个病理样本的特征

# 投影到对比学习空间
mammo_proj, patho_proj = alignment(z_mammo, z_patho)
# 输出都是 [4, 256]，且已经L2归一化

# 计算对比损失
labels = torch.tensor([0, 0, 1, 1])  # 前两个是同一类，后两个是同一类
loss = compute_contrastive_loss(mammo_proj, patho_proj, labels=labels, temperature=0.07)
```

**正负样本对策略：**

对比学习需要正样本对和负样本对。我们的策略是：
- 如果提供了`labels`：同一标签的样本被视为正样本对。比如两个都是"恶性"的样本，它们的X光和病理特征应该相近。
- 如果没有提供`labels`：默认对角线元素是正样本对，即第i个X光样本和第i个病理样本配对。

**实际使用中的问题：**

1. **batch size太小**：对比学习需要足够的负样本，batch size太小（比如<4）效果不好。建议至少8。
2. **温度参数**：如果loss太大或太小，可以调整temperature。太大（>0.1）会让所有样本都相似，太小（<0.01）会让学习变得困难。
3. **特征归一化**：投影后的特征会自动L2归一化，这是对比学习的标准做法。不要手动再归一化。

### compute_contrastive_loss

这是计算对比损失的函数，实现了InfoNCE损失。

**参数说明：**

- `mammo_features`: X光特征，形状[B, D]，必须已经L2归一化。
- `patho_features`: 病理特征，形状[B, D]，必须已经L2归一化。
- `labels`: 可选，形状[B]。用于构建正负样本对。
- `temperature`: 温度参数，默认0.07。

**返回值：**

标量tensor，表示对比损失。

**损失计算过程：**

1. 计算相似度矩阵：`similarity = mammo_features @ patho_features.T`，形状[B, B]
2. 除以temperature得到logits
3. 根据labels或对角线构建正样本对
4. 计算交叉熵损失（InfoNCE）

**调试技巧：**

如果loss一直很大（>5），可能是：
- 特征没有正确归一化
- temperature太小
- 正样本对构建错误

如果loss一直很小（<0.1），可能是：
- temperature太大
- 特征已经对齐得很好（这是好事）

---

## 3. domain_generalization.py - 域泛化

这个模块用来提高模型的域泛化能力，让模型在不同医院、不同设备、不同染色条件下都能工作。包含两个方法：MixStyle和IRM。

### MixStyle

MixStyle通过混合不同样本的统计特征（均值和方差）来模拟域差异。比如不同医院的X光扫描仪可能有不同的对比度，MixStyle可以模拟这种差异。

**初始化参数：**

- `alpha`: Beta分布的参数，控制混合强度，默认0.1。alpha越小，混合越随机；alpha越大，混合越均匀。0.1是一个比较保守的值，不会破坏太多原始特征。
- `p`: 应用MixStyle的概率，默认0.5。只有50%的概率会应用，这样既增加了多样性，又不会完全破坏原始数据。

**使用示例：**

```python
from models.domain_generalization import MixStyle

mixstyle = MixStyle(alpha=0.1, p=0.5)

# 在训练循环中使用
features = some_encoder(x)  # [B, C, H, W]
mixed_features = mixstyle(features)  # 可能混合，也可能不混合（取决于p）
```

**工作原理：**

1. 计算每个样本的均值和方差：`mu = mean(features, dim=[2,3])`, `sigma = std(features, dim=[2,3])`
2. 随机打乱batch顺序
3. 从Beta分布采样混合权重lambda
4. 混合统计量：`mu_mixed = lambda * mu + (1-lambda) * mu_shuffled`
5. 用新统计量重新标准化特征

**注意事项：**

- 只在训练时启用（`model.train()`），推理时自动跳过。
- 如果batch size太小（<4），混合效果不明显。
- alpha不要太大，否则会破坏太多原始特征，影响学习。

### IRMLoss

IRM（Invariant Risk Minimization）是一种域泛化方法，目标是学习在所有域上都最优的特征。

**初始化参数：**

- `penalty_weight`: IRM惩罚项的权重，默认1.0。这个权重控制惩罚项的强度，太大可能让模型学不到东西，太小可能没有效果。

**使用示例：**

```python
from models.domain_generalization import IRMLoss

irm_loss = IRMLoss(penalty_weight=1.0)

# 假设有多个域的数据
logits_domain1 = model(x_domain1)  # 域1的输出
logits_domain2 = model(x_domain2)  # 域2的输出
labels_domain1 = torch.tensor([0, 1, 2, 3])
labels_domain2 = torch.tensor([0, 1, 2, 3])

# 计算IRM损失
total_loss, penalty = irm_loss(
    [logits_domain1, logits_domain2],
    [labels_domain1, labels_domain2]
)
```

**工作原理：**

IRM的核心思想是：如果一个特征表示在所有域上都最优，那么它应该是域不变的。我们通过惩罚损失的方差来实现这一点。如果模型在不同域上的损失差异很大，说明特征不是域不变的，需要惩罚。

**实际使用建议：**

- 需要至少2个域的数据才能用IRM。
- penalty_weight需要仔细调，建议从0.5开始，根据验证集性能调整。
- IRM会增加训练时间，因为要计算多个域的损失。

### DomainGeneralization

这是整合MixStyle和IRM的包装类，方便使用。

```python
from models.domain_generalization import DomainGeneralization

dg = DomainGeneralization(use_mixstyle=True, mixstyle_alpha=0.1, irm_penalty=1.0)

# 应用MixStyle
features = dg.apply_mixstyle(features)

# 计算IRM损失
total_loss, penalty = dg.compute_irm_loss(logits_list, labels_list)
```

---

## 4. causal_tta.py - 测试时自适应

测试时自适应（Test-Time Adaptation, TTA）是在推理时用无标签的目标域数据微调模型。我们的实现是"因果"的，意思是只更新统计参数（BN层），不改变诊断路径。

### CausalTTA

这个类实现了因果测试时自适应。

**初始化参数：**

- `model`: 要自适应的模型。注意模型必须是已经训练好的。
- `lr`: 学习率，默认1e-4。TTA的学习率要很小，因为只是微调，不能破坏源域的性能。
- `entropy_weight`: 熵最小化权重，默认0.1。我们通过最小化预测熵来让模型输出更确定。

**使用示例：**

```python
from models.causal_tta import CausalTTA
from models.breast_cancer_model import BreastCancerDiagnosisModel

# 加载训练好的模型
model = BreastCancerDiagnosisModel()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 创建TTA适配器
tta = CausalTTA(model, lr=1e-4, entropy_weight=0.1)

# 在目标域数据上自适应（无标签）
tta.adapt(target_dataloader, num_steps=100)

# 现在可以用自适应后的模型推理
outputs = model(mammo_img, patho_img)
```

**工作原理：**

1. **只优化BN层和Adapter层**：我们假设域差异主要来自图像风格（对比度、亮度等），而不是病变本身。所以只更新统计参数，不更新诊断相关的权重。
2. **熵最小化**：让模型输出更确定，减少不确定性。
3. **源域约束**：防止BN统计量偏离太远，保持源域性能。

**adapt_step方法：**

这是执行一步自适应的函数。

```python
# 单步自适应
loss = tta.adapt_step(x, source_constraint_weight=0.01)
```

`source_constraint_weight`控制源域约束的强度，默认0.01。如果发现自适应后源域性能下降太多，可以增大这个值。

**注意事项：**

1. **需要目标域数据**：TTA需要一些目标域的无标签数据。通常50-200个样本就够了。
2. **步数不要太多**：num_steps建议50-200，太多可能会过拟合目标域，导致源域性能下降。
3. **学习率要小**：lr建议1e-4或更小，太大可能会破坏模型。
4. **只在推理时用**：训练时不需要TTA。

**常见问题：**

- **Q: TTA后源域性能下降怎么办？**
  A: 增大source_constraint_weight，或者减少num_steps。

- **Q: 需要多少目标域数据？**
  A: 至少50个样本，建议100-200个。太少效果不明显，太多可能过拟合。

- **Q: 可以用于训练吗？**
  A: 不建议。TTA是专门为推理时适应新域设计的，训练时用域泛化方法更好。

---

## 5. multi_task_heads.py - 多任务学习头

这个模块实现了三个任务头：分类、定位、分级。还实现了不确定性加权来平衡多任务学习。

### ClassificationHead

分类头，输出4类：Normal, Benign, In situ, Invasive。

**结构：**
```
输入 [B, 768] 
→ Linear(768, 384) + ReLU + Dropout
→ Linear(384, num_classes)
→ 输出 [B, num_classes]
```

**使用：**

```python
from models.multi_task_heads import ClassificationHead

cls_head = ClassificationHead(embed_dim=768, num_classes=4, dropout=0.1)
features = torch.randn(4, 768)
logits = cls_head(features)  # [4, 4]
probs = F.softmax(logits, dim=1)  # 转换为概率
pred = logits.argmax(dim=1)  # 预测类别
```

### LocalizationHead

定位头，输出边界框坐标 [x, y, w, h]，都是归一化到[0,1]的。

**结构：**
```
输入 [B, 768]
→ Linear(768, 256) + ReLU + Dropout
→ Linear(256, 128) + ReLU
→ Linear(128, 4) + Sigmoid
→ 输出 [B, 4]
```

**坐标格式：**
- x, y: 边界框左上角坐标（归一化）
- w, h: 边界框宽度和高度（归一化）

**使用：**

```python
from models.multi_task_heads import LocalizationHead, compute_bbox_loss

loc_head = LocalizationHead(embed_dim=768, hidden_dim=256, dropout=0.1)
features = torch.randn(4, 768)
bbox_pred = loc_head(features)  # [4, 4]

# 计算损失
bbox_gt = torch.tensor([
    [0.1, 0.1, 0.3, 0.3],  # [x, y, w, h]
    [0.2, 0.2, 0.4, 0.4],
    [0.3, 0.3, 0.2, 0.2],
    [0.4, 0.4, 0.3, 0.3]
])
loss = compute_bbox_loss(bbox_pred, bbox_gt)
```

**compute_bbox_loss：**

这个函数计算边界框损失，包含L1损失和IoU损失。

```python
loss = compute_bbox_loss(pred_bbox, target_bbox)
```

损失 = L1损失 + IoU损失。IoU损失鼓励预测框和真实框重叠更多。

### GradingHead

分级头，用于序数回归。输出3级：Benign(0), In situ(1), Invasive(2)。

**序数回归的特点：**
- 相邻等级的错误惩罚轻（比如预测1但实际是2）
- 远距离等级的错误惩罚重（比如预测0但实际是2）

**使用：**

```python
from models.multi_task_heads import GradingHead, compute_ordinal_loss

grade_head = GradingHead(embed_dim=768, num_grades=3, dropout=0.1)
features = torch.randn(4, 768)
grade_logits = grade_head(features)  # [4, 3]

# 计算序数损失
grade_gt = torch.tensor([0, 1, 2, 1])  # 真实等级
loss = compute_ordinal_loss(grade_logits, grade_gt)
```

**compute_ordinal_loss：**

这个函数实现了序数回归损失。对于每个样本，计算所有可能等级的加权交叉熵，权重根据距离真实等级的距离设置。

### UncertaintyWeightedMultiTaskLoss

这是不确定性加权的多任务损失。核心思想是：不同任务的难度不同，应该给它们不同的权重。我们让模型自动学习这些权重。

**工作原理：**

每个任务有一个可学习的参数σ（实际存储的是log(σ²)）。任务的权重是 1/(2σ²)。如果任务越不确定（σ越大），权重越小，这样模型会更多地关注容易的任务。

**使用：**

```python
from models.multi_task_heads import UncertaintyWeightedMultiTaskLoss

# 创建不确定性加权损失
uncertainty_loss = UncertaintyWeightedMultiTaskLoss(num_tasks=3)

# 计算各任务的损失
loss_cls = F.cross_entropy(cls_logits, labels_cls)
loss_loc = compute_bbox_loss(bbox_pred, bbox_gt)
loss_grade = compute_ordinal_loss(grade_logits, labels_grade)

# 加权
total_loss, task_weights = uncertainty_loss([loss_cls, loss_loc, loss_grade])
print(f"任务权重: {task_weights}")  # 可以看到哪个任务权重高
```

**注意事项：**

- 权重是自动学习的，不需要手动设置。
- 如果某个任务的权重一直很小，可能是这个任务太难了，或者数据有问题。
- 初始时所有任务的权重是1.0，训练过程中会自动调整。

---

## 6. interpretability.py - 可解释性模块

这个模块用来生成可解释性结果，包括热图和病理patch关联。帮助医生理解模型为什么做出某个诊断。

### CrossModalAttention

跨模态注意力机制，计算X光patch到病理patch的注意力。

**工作原理：**

1. X光patch作为Query，病理patch作为Key和Value
2. 计算注意力分数：`attention = softmax(Q @ K.T / sqrt(d))`
3. 用注意力权重加权Value得到输出

**使用：**

```python
from models.interpretability import CrossModalAttention

attention = CrossModalAttention(embed_dim=768, num_heads=8, dropout=0.1)

mammo_tokens = torch.randn(2, 196, 768)  # 196个X光patch
patho_tokens = torch.randn(2, 196, 768)  # 196个病理patch

attended_mammo, attention_weights = attention(mammo_tokens, patho_tokens)
# attended_mammo: [2, 196, 768] - 注意力加权的X光特征
# attention_weights: [2, 196, 196] - 注意力权重矩阵
```

**attention_weights的含义：**
- `attention_weights[i, j, k]` 表示第i个样本的第j个X光patch对第k个病理patch的注意力
- 值越大，说明这个X光patch和这个病理patch越相关

### InterpretabilityModule

这是完整的可解释性模块，可以生成热图和top-k病理patch。

**使用：**

```python
from models.interpretability import InterpretabilityModule

interp_module = InterpretabilityModule(embed_dim=768, num_heads=8, top_k=5)

# 需要先获取patch tokens
_, mammo_tokens = mammo_encoder(mammo_img, return_patch_tokens=True)
_, patho_tokens = patho_encoder(patho_img, return_patch_tokens=True)

# 生成可解释性结果
results = interp_module(mammo_tokens, patho_tokens, img_size=224, patch_size=16)

heatmap = results['heatmap']  # [B, 224, 224] - 热图
top_patches = results['top_patho_patches']  # [B, 5, 768] - top-5病理patch
top_indices = results['top_indices']  # [B, 5] - top-5的索引
attention_weights = results['attention_weights']  # [B, 196, 196] - 完整注意力矩阵
```

**热图生成过程：**

1. 对每个X光patch，计算它对所有病理patch的平均注意力
2. 得到每个X光patch的重要性分数
3. 将patch级别的分数reshape成空间网格
4. 上采样到原图大小
5. 归一化到[0, 1]

**可视化热图：**

```python
import matplotlib.pyplot as plt
import numpy as np

# 假设有原图
mammo_img_np = mammo_img[0, 0].cpu().numpy()  # [224, 224]
heatmap_np = heatmap[0].cpu().numpy()  # [224, 224]

# 叠加显示
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(mammo_img_np, cmap='gray')
plt.title('Original X-ray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mammo_img_np, cmap='gray', alpha=0.5)
plt.imshow(heatmap_np, cmap='hot', alpha=0.5)
plt.title('Heatmap Overlay')
plt.axis('off')
plt.show()
```

**top-k病理patch：**

这个功能可以找出与X光最相关的k个病理patch。比如模型判断某个区域是恶性，可以看看它最关注病理图像的哪些部分。

```python
# 获取top-5病理patch的索引
top5_indices = top_indices[0].cpu().numpy()  # [5]

# 可以根据索引从原始病理图像中提取对应的patch
# 每个patch是16x16的（如果patch_size=16）
```

---

## 7. breast_cancer_model.py - 主模型

这是整合所有模块的完整模型。提供了统一的接口，方便训练和推理。

### BreastCancerDiagnosisModel

**初始化参数：**

- `img_size`: 图像大小，默认224
- `patch_size`: Patch大小，默认16
- `embed_dim`: 嵌入维度，默认768
- `depth`: Transformer深度，默认12
- `num_heads`: 注意力头数，默认12
- `num_classes`: 分类类别数，默认4（Normal, Benign, In situ, Invasive）
- `num_grades`: 分级数量，默认3（Benign, In situ, Invasive）
- `use_dg`: 是否使用域泛化，默认True
- `use_interpretability`: 是否使用可解释性，默认True。如果设为False，forward不会返回heatmap等。
- `projection_dim`: 跨模态对齐的投影维度，默认256
- `temperature`: 对比学习的温度参数，默认0.07

**forward方法：**

```python
outputs = model(mammo_img, patho_img=None, return_interpretability=False, return_patch_tokens=False)
```

**参数：**
- `mammo_img`: [B, 1, H, W] X光图像，必须提供
- `patho_img`: [B, 3, H, W] 病理图像，可选。如果为None，模型只使用X光特征
- `return_interpretability`: 是否返回可解释性结果，会增加计算和内存
- `return_patch_tokens`: 是否返回patch tokens，通常不需要

**返回值（字典）：**
- `classification`: [B, num_classes] 分类logits
- `localization`: [B, 4] 边界框 [x, y, w, h]
- `grading`: [B, num_grades] 分级logits
- `z_mammo`: [B, embed_dim] X光特征（如果return_patch_tokens=True）
- `z_patho`: [B, embed_dim] 病理特征（如果提供了patho_img）
- `mammo_proj`: [B, projection_dim] 投影后的X光特征（如果提供了patho_img）
- `patho_proj`: [B, projection_dim] 投影后的病理特征（如果提供了patho_img）
- `heatmap`: [B, H, W] 热图（如果return_interpretability=True）
- `top_patho_patches`: [B, top_k, embed_dim] top-k病理patch（如果return_interpretability=True）

**compute_loss方法：**

```python
total_loss, loss_dict = model.compute_loss(
    outputs,
    labels_cls=None,
    labels_bbox=None,
    labels_grade=None,
    labels_align=None,
    mammo_features=None,
    patho_features=None
)
```

**参数说明：**
- `outputs`: forward的输出字典
- `labels_cls`: [B] 分类标签，0-3
- `labels_bbox`: [B, 4] 边界框标签，归一化坐标
- `labels_grade`: [B] 分级标签，0-2
- `labels_align`: [B] 对齐标签，用于对比学习，通常和labels_cls相同
- `mammo_features`: [B, D] X光特征（用于对齐），通常是outputs['mammo_proj']
- `patho_features`: [B, D] 病理特征（用于对齐），通常是outputs['patho_proj']

**返回值：**
- `total_loss`: 总损失（标量tensor）
- `loss_dict`: 字典，包含各损失项：
  - `classification`: 分类损失
  - `localization`: 定位损失
  - `grading`: 分级损失
  - `alignment`: 对齐损失
  - `total`: 总损失
  - `task_weights`: 各任务的权重（列表）

**完整训练示例：**

```python
from models.breast_cancer_model import BreastCancerDiagnosisModel
import torch
import torch.nn.functional as F

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
    use_interpretability=False  # 训练时通常不需要可解释性
)

# 准备数据
mammo_img = torch.randn(4, 1, 224, 224)
patho_img = torch.randn(4, 3, 224, 224)
labels_cls = torch.tensor([0, 1, 2, 3])
labels_bbox = torch.randn(4, 4).clamp(0, 1)
labels_grade = torch.tensor([0, 1, 2, 1])

# 前向传播
outputs = model(mammo_img, patho_img)

# 计算损失
total_loss, loss_dict = model.compute_loss(
    outputs,
    labels_cls=labels_cls,
    labels_bbox=labels_bbox,
    labels_grade=labels_grade,
    labels_align=labels_cls,
    mammo_features=outputs['mammo_proj'],
    patho_features=outputs['patho_proj']
)

# 反向传播
total_loss.backward()
optimizer.step()
optimizer.zero_grad()

# 打印损失
print(f"总损失: {total_loss.item():.4f}")
print(f"分类损失: {loss_dict['classification'].item():.4f}")
print(f"定位损失: {loss_dict['localization'].item():.4f}")
print(f"分级损失: {loss_dict['grading'].item():.4f}")
print(f"对齐损失: {loss_dict['alignment'].item():.4f}")
```

**推理示例：**

```python
# 加载模型
model = BreastCancerDiagnosisModel()
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    outputs = model(mammo_img, patho_img, return_interpretability=True)
    
    # 获取预测
    pred_class = outputs['classification'].argmax(dim=1)
    pred_bbox = outputs['localization']
    pred_grade = outputs['grading'].argmax(dim=1)
    heatmap = outputs['heatmap']
    
    # 转换为概率
    probs = F.softmax(outputs['classification'], dim=1)
    print(f"预测类别: {pred_class}")
    print(f"预测概率: {probs.max(dim=1)[0]}")
```

**常见问题：**

1. **只有X光图像怎么办？**
   ```python
   outputs = model(mammo_img, patho_img=None)
   # 模型会自动只使用X光特征
   ```

2. **内存不够怎么办？**
   - 减小batch size
   - 设置`use_interpretability=False`
   - 减小模型大小（embed_dim, depth）

3. **训练很慢怎么办？**
   - 使用混合精度训练（torch.cuda.amp）
   - 减小模型深度
   - 使用更少的epoch

4. **损失不下降怎么办？**
   - 检查数据是否正确加载
   - 检查标签是否正确
   - 调整学习率
   - 检查梯度是否正常（可以用torch.nn.utils.clip_grad_norm_）

---

## 8. __init__.py - 模块导出

这个文件统一导出所有模块，方便使用。

```python
from models import (
    BreastCancerDiagnosisModel,
    SharedViTEncoder,
    CrossModalAlignment,
    compute_contrastive_loss,
    DomainGeneralization,
    MixStyle,
    IRMLoss,
    CausalTTA,
    ClassificationHead,
    LocalizationHead,
    GradingHead,
    UncertaintyWeightedMultiTaskLoss,
    InterpretabilityModule,
    CrossModalAttention
)
```

---

## 实际使用建议

### 模型大小选择

根据你的计算资源选择：

- **小模型**（快速测试，GPU内存<8GB）:
  ```python
  embed_dim=384, depth=6, num_heads=6
  ```

- **中等模型**（推荐，GPU内存8-16GB）:
  ```python
  embed_dim=768, depth=12, num_heads=12
  ```

- **大模型**（最佳性能，GPU内存>16GB）:
  ```python
  embed_dim=1024, depth=24, num_heads=16
  ```

### 训练参数

- **学习率**: 1e-4（AdamW），如果loss震荡可以降到5e-5
- **批次大小**: 8-16，根据GPU内存调整
- **权重衰减**: 1e-4
- **梯度裁剪**: 建议使用，`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

### 数据准备

- X光图像：单通道，归一化到[-1, 1]
- 病理图像：RGB三通道，用ImageNet统计量归一化
- 标签：从0开始，分类0-3，分级0-2

### 调试技巧

1. **先测试小模型**：用embed_dim=384, depth=6快速验证代码是否正确
2. **检查数据加载**：打印几个样本看看图像和标签是否正确
3. **监控损失**：如果某个损失一直很大，可能是数据或代码有问题
4. **可视化中间结果**：可以保存一些中间特征看看是否正常

### 常见错误

1. **维度不匹配**：检查输入图像的通道数，X光必须是1通道，病理必须是3通道
2. **CUDA out of memory**：减小batch size或模型大小
3. **NaN损失**：可能是学习率太大，或者数据有问题
4. **训练不收敛**：检查数据加载、标签、学习率

---

## 总结

这个models目录包含了完整的跨模态乳腺癌诊断系统。每个模块都是独立的，可以单独使用，也可以组合使用。建议先从主模型`BreastCancerDiagnosisModel`开始，它已经整合了所有功能。如果需要对某个模块进行定制，可以单独导入使用。

如果有问题，可以查看各个文件的代码注释，或者参考论文中的方法部分。
