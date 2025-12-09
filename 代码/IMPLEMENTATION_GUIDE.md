# 跨模态乳腺癌诊断系统实现指南

## 项目结构

```
乳腺癌/
├── models/                    # 模型模块
│   ├── __init__.py
│   ├── shared_encoder.py      # 共享ViT编码器（带Adapter）
│   ├── cross_modal_alignment.py  # 跨模态对齐模块
│   ├── domain_generalization.py  # 域泛化（MixStyle + IRM）
│   ├── causal_tta.py          # 因果测试时自适应
│   ├── multi_task_heads.py    # 多任务学习头
│   ├── interpretability.py   # 可解释性模块
│   └── breast_cancer_model.py # 主模型类
├── data/
│   └── dataset.py             # 数据加载器
├── train.py                   # 训练脚本
├── example_usage.py           # 使用示例
└── IMPLEMENTATION_GUIDE.md    # 本文档
```

## 核心模块说明

### 1. 共享编码器（Shared Encoder）

**文件**: `models/shared_encoder.py`

- **SharedViTEncoder**: 统一的视觉Transformer编码器
  - 支持X光（单通道）和病理（RGB）两种模态
  - 使用Adapter模块保留模态特定特征
  - 输出统一的嵌入特征表示

**关键特性**:
- Patch embedding: 将图像分割成16×16的patches
- Positional embedding: 位置编码
- Modality embedding: 模态特定嵌入（区分X光和病理）
- Transformer blocks: 多层Transformer编码块
- Adapter layers: 轻量级适配器，防止模态特征过度混合

### 2. 跨模态对齐（Cross-modal Alignment）

**文件**: `models/cross_modal_alignment.py`

- **CrossModalAlignment**: 跨模态对齐模块
  - 将X光和病理特征投影到对比学习空间
  - 使用InfoNCE损失进行对比学习

**训练策略**:
- 同一病人的X光和病理 → 正样本对
- 不同病人或不同诊断 → 负样本对
- 弱监督学习（无需像素级配对）

### 3. 域泛化（Domain Generalization）

**文件**: `models/domain_generalization.py`

- **MixStyle**: 混合不同域的统计特征（均值和方差）
  - 模拟不同扫描仪、染色风格
  - 增强模型对域差异的鲁棒性

- **IRMLoss**: Invariant Risk Minimization损失
  - 约束模型在所有域上都达到最优
  - 学习域不变特征

### 4. 测试时自适应（Causal TTA）

**文件**: `models/causal_tta.py`

- **CausalTTA**: 因果测试时自适应
  - 只更新BN层统计参数和Adapter权重
  - 保持诊断路径不变
  - 最小化预测熵，使输出更稳定
  - 约束源域统计量，防止崩塌

**使用场景**: 推理时遇到新医院/设备的数据

### 5. 多任务学习头（Multi-task Heads）

**文件**: `models/multi_task_heads.py`

- **ClassificationHead**: 四类分类（Normal, Benign, In situ, Invasive）
- **LocalizationHead**: 病灶定位（边界框回归）
- **GradingHead**: 病理分级（序数回归）

**损失函数**:
- 分类: 交叉熵
- 定位: L1 + IoU损失
- 分级: 序数回归损失（相邻等级惩罚轻，远距离惩罚重）

**不确定性加权**: 使用可学习的权重参数σ动态平衡多任务

### 6. 可解释性模块（Interpretability）

**文件**: `models/interpretability.py`

- **CrossModalAttention**: 跨模态注意力机制
  - X光patch → 病理patch的注意力
  - 生成注意力权重矩阵

- **InterpretabilityModule**: 可解释性模块
  - 从注意力权重生成热图
  - 提取top-k最相关的病理patch
  - 可视化模型决策依据

## 数据格式

### X光图像（Mammography）
- 格式: DICOM (.dcm)
- 通道: 单通道（灰度）
- 预处理:
  - 窗宽窗位调整
  - 直方图均衡化
  - 归一化到[-1, 1]

### 病理图像（Histopathology）
- 格式: PNG/JPEG
- 通道: RGB（3通道）
- 预处理:
  - Macenko染色归一化
  - 归一化到ImageNet统计量

### 标注格式
CSV文件应包含以下列：
- `image file path`: X光图像路径
- `ROI mask file path`: ROI mask路径（可选）
- `pathology` 或 `assessment`: 诊断标签（BENIGN/MALIGNANT）
- `patient_id`: 病人ID（用于配对X光和病理）

## 训练流程

### 1. 准备数据

```bash
# 组织数据目录
data/
├── mammography/
│   └── ... (DICOM files)
├── pathology/
│   └── ... (PNG/JPEG files)
└── annotations.csv
```

### 2. 运行训练

```bash
python train.py \
    --mammo_csv data/annotations.csv \
    --mammo_root data/mammography \
    --patho_root data/pathology \
    --use_pathology \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints \
    --log_dir ./logs
```

### 3. 监控训练

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir ./logs
```

## 推理流程

### 1. 加载模型

```python
from models import BreastCancerDiagnosisModel

model = BreastCancerDiagnosisModel(
    img_size=224,
    patch_size=16,
    embed_dim=768
)

checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 2. 单样本推理

```python
# 加载图像
mammo_img = load_mammography_image('path/to/image.dcm')
patho_img = load_pathology_image('path/to/pathology.png')

# 推理
with torch.no_grad():
    outputs = model(mammo_img, patho_img, return_interpretability=True)

# 获取结果
pred_class = outputs['classification'].argmax(dim=1)
pred_bbox = outputs['localization']
pred_grade = outputs['grading'].argmax(dim=1)
heatmap = outputs['heatmap']
```

### 3. 测试时自适应（新域）

```python
from models.causal_tta import CausalTTA

# 创建TTA适配器
tta = CausalTTA(model, lr=1e-4)

# 在目标域数据上自适应
tta.adapt(target_dataloader, num_steps=100)

# 推理
outputs = model(mammo_img, patho_img)
```

## 关键设计决策

### 1. 为什么使用共享编码器？
- 统一特征空间，便于跨模态对齐
- 减少参数量，提高训练效率
- Adapter模块保留模态特定特征

### 2. 为什么使用弱监督对齐？
- 公共数据集没有像素级配对
- 同一病人的X光和病理可作为正样本对
- 对比学习可以自动学习语义对应关系

### 3. 为什么使用MixStyle + IRM？
- MixStyle: 增强对风格差异的鲁棒性
- IRM: 学习域不变特征
- 两者结合：全面的域泛化能力

### 4. 为什么使用因果TTA？
- 只更新统计参数，不改变诊断逻辑
- 适应新域的同时保持源域性能
- 无监督自适应，无需目标域标注

### 5. 为什么使用不确定性加权？
- 不同任务难度不同
- 动态平衡多任务学习
- 避免简单任务主导训练

## 性能优化建议

1. **混合精度训练**: 使用`torch.cuda.amp`加速训练
2. **梯度累积**: 小batch size时使用梯度累积
3. **数据并行**: 多GPU训练使用`DataParallel`或`DistributedDataParallel`
4. **模型剪枝**: 推理时可以使用模型剪枝减少参数量

## 常见问题

### Q: 如何处理没有病理图像的情况？
A: 设置`use_pathology=False`，模型只使用X光特征进行预测。

### Q: 如何调整多任务权重？
A: 模型使用不确定性加权，权重会自动学习。也可以手动调整`UncertaintyWeightedMultiTaskLoss`中的初始值。

### Q: TTA需要多少步？
A: 通常50-200步即可，可以通过验证集性能确定最优步数。

### Q: 如何可视化热图？
A: 使用`outputs['heatmap']`，可以叠加在原图上进行可视化。

## 引用

如果使用本代码，请引用相关论文。

