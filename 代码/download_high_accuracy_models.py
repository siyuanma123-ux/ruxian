"""
下载和使用高准确率开源乳腺癌诊断模型
目标：准确率80%以上
"""

import os
import sys
import subprocess
from pathlib import Path

# 高准确率模型列表
HIGH_ACCURACY_MODELS = {
    "1": {
        "name": "End-to-end All Convolutional Design",
        "paper": "https://arxiv.org/abs/1711.05775",
        "github": "https://github.com/lishen/end2end-all-conv",
        "accuracy": "AUC 0.91 (DDSM), 0.95 (INbreast)",
        "description": "端到端全卷积网络，在DDSM和INbreast数据集上表现优异",
        "download_cmd": "git clone https://github.com/lishen/end2end-all-conv.git"
    },
    "2": {
        "name": "Deep Learning for Early Breast Cancer Detection",
        "paper": "https://arxiv.org/abs/1708.09427",
        "github": "需要查找具体仓库",
        "accuracy": "AUC 0.91-0.98, 敏感性86.7%, 特异性96.1%",
        "description": "全卷积网络方法，在INbreast数据库上AUC达到0.95-0.98",
        "download_cmd": None
    },
    "3": {
        "name": "EfficientNet Dual-View Mammography",
        "paper": "https://arxiv.org/abs/2110.01606",
        "github": "需要查找具体仓库",
        "accuracy": "AUC 0.9344, 准确率85.13%",
        "description": "基于EfficientNet的双视图X光片诊断模型，在CBIS-DDSM上训练",
        "download_cmd": None
    },
    "4": {
        "name": "MV-Swin-T (Multi-view Swin Transformer)",
        "paper": "https://arxiv.org/abs/2402.16298",
        "github": "https://github.com/prithuls/MV-Swin-T",
        "accuracy": "在CBIS-DDSM和Vin-Dr Mammo上表现优异",
        "description": "多视图Swin Transformer，专门为乳腺X光片分类设计",
        "download_cmd": "git clone https://github.com/prithuls/MV-Swin-T.git"
    },
    "5": {
        "name": "Hybrid Transfer Learning (MVGG16)",
        "paper": "https://arxiv.org/abs/2003.13503",
        "github": "需要查找具体仓库",
        "accuracy": "准确率88.3% (DDSM)",
        "description": "融合改进VGG16和ImageNet的混合迁移学习模型",
        "download_cmd": None
    }
}


def print_model_list():
    """打印模型列表"""
    print("=" * 80)
    print("高准确率开源乳腺癌诊断模型 (准确率≥80%)")
    print("=" * 80)
    print()
    
    for key, model in HIGH_ACCURACY_MODELS.items():
        print(f"【{key}】{model['name']}")
        print(f"  准确率: {model['accuracy']}")
        print(f"  描述: {model['description']}")
        print(f"  论文: {model['paper']}")
        if model['github'] != "需要查找具体仓库":
            print(f"  GitHub: {model['github']}")
        if model['download_cmd']:
            print(f"  可下载: ✅")
        else:
            print(f"  可下载: ⚠️  需要手动查找")
        print()


def download_model(model_key, target_dir="pretrained_models"):
    """下载指定模型"""
    if model_key not in HIGH_ACCURACY_MODELS:
        print(f"错误: 模型编号 {model_key} 不存在")
        return False
    
    model = HIGH_ACCURACY_MODELS[model_key]
    
    if not model['download_cmd']:
        print(f"⚠️  模型 '{model['name']}' 没有直接的下载命令")
        print(f"   请访问论文页面查找GitHub链接: {model['paper']}")
        return False
    
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在下载: {model['name']}")
    print(f"目标目录: {target_dir}")
    print()
    
    try:
        # 执行git clone
        cmd = model['download_cmd'].split()
        cmd.append(str(target_dir / model['name'].lower().replace(' ', '_')))
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ 下载成功!")
        print(f"模型已保存到: {target_dir / model['name'].lower().replace(' ', '_')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        return False


def create_integration_guide():
    """创建集成指南"""
    guide_content = """# 高准确率模型集成指南

## 推荐模型（准确率≥80%）

### 1. End-to-end All Convolutional Design ⭐ 推荐
- **准确率**: AUC 0.91 (DDSM), 0.95 (INbreast)
- **GitHub**: https://github.com/lishen/end2end-all-conv
- **特点**: 
  - 端到端训练
  - 全卷积网络设计
  - 有公开代码和预训练权重

**使用方法**:
```bash
git clone https://github.com/lishen/end2end-all-conv
cd end2end-all-conv
# 按照README安装依赖
# 下载预训练权重
python inference.py --image your_image.png
```

### 2. MV-Swin-T (Multi-view Swin Transformer)
- **准确率**: 在CBIS-DDSM和Vin-Dr Mammo上表现优异
- **GitHub**: https://github.com/prithuls/MV-Swin-T
- **特点**:
  - 多视图信息融合
  - 基于Swin Transformer
  - 专门为乳腺X光片设计

**使用方法**:
```bash
git clone https://github.com/prithuls/MV-Swin-T
cd MV-Swin-T
# 按照README安装依赖和下载权重
```

### 3. EfficientNet Dual-View
- **准确率**: AUC 0.9344, 准确率85.13%
- **论文**: https://arxiv.org/abs/2110.01606
- **特点**: 双视图X光片诊断

**注意**: 需要查找具体GitHub仓库

## 集成到对比脚本

### 方法1: 直接调用模型推理

```python
# 在 compare_with_baseline_models.py 中添加
import sys
sys.path.append('path/to/end2end-all-conv')

from model import YourModel

class EndToEndModel(nn.Module):
    def __init__(self, num_classes=4):
        # 加载预训练模型
        self.model = load_pretrained_model('path/to/weights.pth')
    
    def forward(self, x):
        return self.model(x)
```

### 方法2: 使用模型API

如果模型提供了API接口：
```python
import requests

def diagnose_with_api(image_path):
    # 调用模型API
    response = requests.post(
        'model_api_url',
        files={'image': open(image_path, 'rb')}
    )
    return response.json()
```

## 注意事项

1. **数据格式**: 确保输入图像格式与模型要求一致
2. **预处理**: 不同模型可能需要不同的预处理方式
3. **类别映射**: 注意不同模型的类别定义可能不同
4. **权重下载**: 某些模型需要单独下载预训练权重
5. **依赖环境**: 注意Python版本和依赖库版本

## 性能对比

建议使用以下指标进行对比：
- 准确率 (Accuracy)
- AUC (Area Under Curve)
- 敏感性 (Sensitivity)
- 特异性 (Specificity)
- F1-Score

## 下一步

1. 下载推荐的模型代码
2. 安装依赖环境
3. 下载预训练权重
4. 测试模型在您的数据上的表现
5. 集成到对比脚本中
"""
    
    guide_path = Path("高准确率模型集成指南.md")
    with open(guide_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"✅ 集成指南已创建: {guide_path}")
    return guide_path


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='下载高准确率乳腺癌诊断模型')
    parser.add_argument('--list', action='store_true', help='列出所有可用模型')
    parser.add_argument('--download', type=str, help='下载指定模型 (输入模型编号)')
    parser.add_argument('--target_dir', type=str, default='pretrained_models', 
                       help='下载目标目录')
    parser.add_argument('--guide', action='store_true', help='生成集成指南')
    
    args = parser.parse_args()
    
    if args.list:
        print_model_list()
    elif args.download:
        download_model(args.download, args.target_dir)
    elif args.guide:
        create_integration_guide()
    else:
        print("使用方法:")
        print("  --list         列出所有可用模型")
        print("  --download N   下载模型编号N")
        print("  --guide        生成集成指南")
        print()
        print("示例:")
        print("  python download_high_accuracy_models.py --list")
        print("  python download_high_accuracy_models.py --download 1")
        print("  python download_high_accuracy_models.py --guide")
        print()
        print_model_list()


if __name__ == '__main__':
    main()


