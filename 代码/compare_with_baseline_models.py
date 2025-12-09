"""
对比诊断脚本
使用多个开源预训练模型进行诊断，并与我们的模型结果对比
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加路径
sys.path.append(str(Path(__file__).parent))

from models.breast_cancer_model import BreastCancerDiagnosisModel


# 类别名称
CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "良性 (Benign)",
    2: "原位癌 (In-situ)",
    3: "浸润癌 (Invasive)"
}


class BaselineResNet(nn.Module):
    """基于ResNet的baseline模型（使用ImageNet预训练权重）"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        import torchvision.models as models
        
        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=True)
        
        # 修改第一层以接受单通道输入
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 移除最后的分类层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 添加新的分类层
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BaselineEfficientNet(nn.Module):
    """基于EfficientNet的baseline模型"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        try:
            import torchvision.models as models
            # 尝试加载EfficientNet
            efficientnet = models.efficientnet_b0(pretrained=True)
            
            # 修改第一层
            efficientnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
            # 修改分类层
            efficientnet.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(efficientnet.classifier[1].in_features, num_classes)
            )
            
            self.model = efficientnet
        except:
            # 如果EfficientNet不可用，使用ResNet
            print("EfficientNet不可用，使用ResNet替代")
            self.model = BaselineResNet(num_classes)
    
    def forward(self, x):
        return self.model(x)


class BaselineViT(nn.Module):
    """基于Vision Transformer的baseline模型（使用timm库）"""
    
    def __init__(self, num_classes=4, img_size=224):
        super().__init__()
        try:
            import timm
            
            # 加载预训练的ViT
            self.model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=num_classes,
                in_chans=1  # 单通道输入
            )
            self.use_3ch = False
        except ImportError:
            print("timm库未安装，使用ResNet替代")
            self.model = BaselineResNet(num_classes)
            self.use_3ch = False
        except Exception as e:
            print(f"加载ViT失败: {e}，使用ResNet替代")
            self.model = BaselineResNet(num_classes)
            self.use_3ch = False
    
    def forward(self, x):
        # 如果fallback到ResNet，不需要复制通道
        if self.use_3ch and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)


def load_mammography_image(image_path, img_size=224):
    """加载X光图像"""
    try:
        img = Image.open(image_path).convert('L')
        img = img.resize((img_size, img_size), Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32)
        
        # 归一化到[-1, 1]
        img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
        img_array = np.clip(img_array, -3, 3)
        img_array = (img_array + 3) / 6 * 2 - 1
        
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return img_tensor, img_array
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None, None


def diagnose_with_model(model, img_tensor, device, model_name):
    """使用指定模型进行诊断"""
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        if model_name == "我们的模型":
            # 我们的模型需要特殊处理
            outputs = model(img_tensor, patho_img=None, return_interpretability=False)
            logits = outputs['classification']
        else:
            # baseline模型
            logits = model(img_tensor)
    
    probs = F.softmax(logits, dim=1)[0]
    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()
    
    return {
        'pred_class': pred_class,
        'confidence': confidence,
        'probabilities': probs.cpu().numpy().tolist(),
        'class_name': CLASS_NAMES[pred_class]
    }


def compare_models(mammo_dir, output_dir):
    """对比多个模型的诊断结果"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化所有模型
    print("\n初始化模型...")
    models_dict = {}
    
    # 1. 我们的模型
    print("  1. 加载我们的模型...")
    our_model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=4,
        num_grades=3,
        use_dg=True,
        use_interpretability=False  # 对比时不需要可解释性
    ).to(device)
    models_dict["我们的模型"] = our_model
    
    # 2. ResNet50 baseline
    print("  2. 加载ResNet50 baseline...")
    resnet_model = BaselineResNet(num_classes=4).to(device)
    models_dict["ResNet50 (ImageNet预训练)"] = resnet_model
    
    # 3. EfficientNet baseline
    print("  3. 加载EfficientNet baseline...")
    efficientnet_model = BaselineEfficientNet(num_classes=4).to(device)
    models_dict["EfficientNet (ImageNet预训练)"] = efficientnet_model
    
    # 4. ViT baseline
    print("  4. 加载Vision Transformer baseline...")
    vit_model = BaselineViT(num_classes=4).to(device)
    models_dict["ViT (ImageNet预训练)"] = vit_model
    
    print(f"\n共加载 {len(models_dict)} 个模型")
    
    # 遍历所有病人
    mammo_dir = Path(mammo_dir)
    patient_dirs = [d for d in mammo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n找到 {len(patient_dirs)} 个病人文件夹")
    print("=" * 80)
    
    all_comparison_results = []
    
    for patient_dir in tqdm(patient_dirs, desc="处理病人"):
        # 解析病人信息
        dir_name = patient_dir.name
        if '-' in dir_name:
            parts = dir_name.split('-')
            patient_name = parts[0]
            side = parts[1] if len(parts) > 1 else "未知"
        else:
            patient_name = dir_name
            side = "未知"
        
        print(f"\n处理: {patient_name} - {side}")
        
        # 查找图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(patient_dir.glob(ext))
        
        if not image_files:
            print(f"  未找到图像文件，跳过")
            continue
        
        image_files = sorted(image_files)
        print(f"  找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        for img_idx, img_path in enumerate(image_files):
            # 加载图像
            img_tensor, img_array = load_mammography_image(img_path)
            if img_tensor is None:
                continue
            
            # 使用所有模型进行诊断
            results = {}
            for model_name, model in models_dict.items():
                try:
                    result = diagnose_with_model(model, img_tensor, device, model_name)
                    results[model_name] = result
                except Exception as e:
                    print(f"  {model_name} 诊断失败: {e}")
                    results[model_name] = None
            
            # 保存对比结果
            comparison_result = {
                'patient_name': patient_name,
                'side': side,
                'image_path': str(img_path),
                'image_index': img_idx + 1,
                'results': {}
            }
            
            for model_name, result in results.items():
                if result is not None:
                    comparison_result['results'][model_name] = {
                        'predicted_class': result['pred_class'],
                        'class_name': result['class_name'],
                        'confidence': result['confidence'],
                        'all_probabilities': {
                            CLASS_NAMES[i]: prob 
                            for i, prob in enumerate(result['probabilities'])
                        }
                    }
            
            all_comparison_results.append(comparison_result)
    
    # 生成对比报告
    print("\n" + "=" * 80)
    print("生成对比报告...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("多模型诊断对比报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append(f"对比模型数量: {len(models_dict)}")
    report_lines.append(f"处理病人数量: {len(patient_dirs)}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # 按病人分组
    patients_dict = {}
    for result in all_comparison_results:
        key = f"{result['patient_name']}_{result['side']}"
        if key not in patients_dict:
            patients_dict[key] = []
        patients_dict[key].append(result)
    
    # 生成每个病人的对比报告
    for patient_key, patient_results in patients_dict.items():
        parts = patient_key.split('_')
        patient_name = parts[0]
        side = '_'.join(parts[1:]) if len(parts) > 1 else "未知"
        
        report_lines.append(f"【病人: {patient_name} - {side}】")
        report_lines.append("")
        
        for img_result in patient_results:
            report_lines.append(f"  图像 {img_result['image_index']}: {Path(img_result['image_path']).name}")
            report_lines.append("")
            
            # 对比表格
            report_lines.append("  模型诊断结果对比:")
            report_lines.append("  " + "-" * 76)
            report_lines.append(f"  {'模型名称':<30} {'预测类别':<20} {'置信度':<10}")
            report_lines.append("  " + "-" * 76)
            
            for model_name, result_data in img_result['results'].items():
                class_name = result_data['class_name']
                confidence = result_data['confidence']
                report_lines.append(f"  {model_name:<30} {class_name:<20} {confidence*100:>6.2f}%")
            
            report_lines.append("  " + "-" * 76)
            report_lines.append("")
            
            # 详细概率分布
            report_lines.append("  各类别概率分布:")
            for class_idx, class_name in CLASS_NAMES.items():
                report_lines.append(f"    {class_name}:")
                for model_name, result_data in img_result['results'].items():
                    prob = result_data['all_probabilities'].get(class_name, 0.0)
                    bar_length = int(prob * 30)
                    bar = "█" * bar_length + "░" * (30 - bar_length)
                    report_lines.append(f"      {model_name:<30} {prob*100:>6.2f}% {bar}")
                report_lines.append("")
            
            report_lines.append("-" * 80)
            report_lines.append("")
    
    # 统计一致性分析
    report_lines.append("【模型一致性分析】")
    report_lines.append("")
    
    agreement_stats = {}
    for result in all_comparison_results:
        if len(result['results']) < 2:
            continue
        
        # 获取所有模型的预测
        predictions = [
            (name, data['predicted_class']) 
            for name, data in result['results'].items()
        ]
        
        # 统计一致预测
        pred_classes = [pred for _, pred in predictions]
        most_common = max(set(pred_classes), key=pred_classes.count)
        agreement_count = pred_classes.count(most_common)
        agreement_ratio = agreement_count / len(pred_classes)
        
        if agreement_ratio not in agreement_stats:
            agreement_stats[agreement_ratio] = 0
        agreement_stats[agreement_ratio] += 1
    
    report_lines.append("  预测一致性统计:")
    for ratio in sorted(agreement_stats.keys(), reverse=True):
        count = agreement_stats[ratio]
        percentage = ratio * 100
        report_lines.append(f"    {percentage:>5.1f}% 一致性: {count} 张图像")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("重要提示:")
    report_lines.append("1. ResNet50、EfficientNet、ViT使用的是ImageNet预训练权重，")
    report_lines.append("   并非专门为医学影像训练，结果仅供参考。")
    report_lines.append("2. 我们的模型目前使用未训练的权重，结果不可用于临床诊断。")
    report_lines.append("3. 所有模型结果仅供参考，不能替代专业医生诊断。")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = output_dir / "多模型对比诊断报告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 保存JSON
    json_path = output_dir / "多模型对比诊断报告.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': list(models_dict.keys()),
            'total_patients': len(patients_dict),
            'total_images': len(all_comparison_results),
            'comparison_results': all_comparison_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 对比报告已生成:")
    print(f"  文本报告: {report_path}")
    print(f"  JSON报告: {json_path}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='多模型诊断对比')
    parser.add_argument('--mammo_dir', type=str, 
                       default='../钼靶报告',
                       help='钼靶报告文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default='../未命名文件夹 2',
                       help='输出文件夹路径')
    
    args = parser.parse_args()
    
    compare_models(args.mammo_dir, args.output_dir)

