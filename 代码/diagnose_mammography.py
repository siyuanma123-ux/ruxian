"""
乳腺X光片诊断脚本
处理钼靶报告文件夹中的图像，生成详细诊断报告
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


# 类别名称映射
CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "良性 (Benign)",
    2: "原位癌 (In-situ Carcinoma)",
    3: "浸润癌 (Invasive Carcinoma)"
}

GRADE_NAMES = {
    0: "轻度 (Mild)",
    1: "中度 (Moderate)",
    2: "重度 (Severe)"
}

SIDE_NAMES = {
    "左侧": "Left Breast",
    "右侧": "Right Breast"
}


def load_mammography_image(image_path, img_size=224):
    """
    加载X光图像（PNG格式）
    
    Args:
        image_path: 图像路径
        img_size: 目标图像大小
    
    Returns:
        torch.Tensor: 预处理后的图像 [1, 1, H, W]
    """
    try:
        # 加载图像
        img = Image.open(image_path).convert('L')  # 转为灰度图
        
        # Resize
        img = img.resize((img_size, img_size), Image.LANCZOS)
        
        # 转为numpy数组
        img_array = np.array(img, dtype=np.float32)
        
        # 归一化到[-1, 1]
        img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
        img_array = np.clip(img_array, -3, 3)  # 限制范围
        img_array = (img_array + 3) / 6 * 2 - 1  # 映射到[-1, 1]
        
        # 转为tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        return img_tensor, img_array
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None, None


def generate_heatmap(attention_weights, img_size=224):
    """
    生成热图
    
    Args:
        attention_weights: 注意力权重 [num_patches]
        img_size: 图像大小
    
    Returns:
        np.ndarray: 热图 [H, W]
    """
    patch_size = 16
    num_patches_per_side = img_size // patch_size
    
    # Reshape到2D
    if len(attention_weights.shape) == 1:
        heatmap_2d = attention_weights[:num_patches_per_side * num_patches_per_side].reshape(
            num_patches_per_side, num_patches_per_side
        )
    else:
        heatmap_2d = attention_weights[:num_patches_per_side, :num_patches_per_side]
    
    # 上采样到原图大小（使用简单的插值）
    try:
        from scipy.ndimage import zoom
        zoom_factor = img_size / num_patches_per_side
        heatmap = zoom(heatmap_2d, zoom_factor, order=1)
    except ImportError:
        # 如果没有scipy，使用简单的重复
        from PIL import Image
        heatmap_pil = Image.fromarray(heatmap_2d)
        heatmap_pil = heatmap_pil.resize((img_size, img_size), Image.BILINEAR)
        heatmap = np.array(heatmap_pil)
    
    # 归一化到[0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    else:
        heatmap = np.ones_like(heatmap) * 0.5
    
    return heatmap


def visualize_diagnosis(image_path, image_array, outputs, save_path):
    """
    可视化诊断结果
    
    Args:
        image_path: 原始图像路径
        image_array: 图像数组
        outputs: 模型输出
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 原始图像
    ax1 = axes[0]
    ax1.imshow(image_array, cmap='gray')
    ax1.set_title('原始X光图像', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 如果有定位结果，绘制边界框
    if 'localization' in outputs:
        bbox = outputs['localization'][0].cpu().numpy()
        if bbox.sum() > 0:  # 如果有检测到病灶
            x1, y1, x2, y2 = bbox
            h, w = image_array.shape
            rect = patches.Rectangle(
                (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax1.add_patch(rect)
            ax1.text(x1 * w, y1 * h - 5, '检测病灶区域', 
                    color='red', fontsize=10, fontweight='bold')
    
    # 热图叠加
    ax2 = axes[1]
    ax2.imshow(image_array, cmap='gray', alpha=0.7)
    
    if 'heatmap' in outputs:
        heatmap = outputs['heatmap'][0].cpu().numpy()
        im = ax2.imshow(heatmap, cmap='hot', alpha=0.5, interpolation='bilinear')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_title('注意力热图 (红色区域为可疑病灶)', fontsize=14, fontweight='bold')
    else:
        ax2.set_title('诊断结果可视化', fontsize=14, fontweight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_diagnosis_report(patient_name, side, image_paths, outputs_list, save_dir):
    """
    生成详细诊断报告
    
    Args:
        patient_name: 病人姓名
        side: 左右侧
        image_paths: 图像路径列表
        outputs_list: 模型输出列表
        save_dir: 保存目录
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"乳腺X光诊断报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"病人姓名: {patient_name}")
    report_lines.append(f"检查部位: {SIDE_NAMES.get(side, side)}")
    report_lines.append(f"检查日期: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append(f"图像数量: {len(image_paths)}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    # 汇总统计
    all_classes = []
    all_grades = []
    all_confidences = []
    all_bboxes = []
    
    for idx, (img_path, outputs) in enumerate(zip(image_paths, outputs_list)):
        report_lines.append(f"【图像 {idx + 1}】")
        report_lines.append(f"文件路径: {img_path}")
        report_lines.append("")
        
        # 分类结果
        if 'classification' in outputs:
            probs = F.softmax(outputs['classification'], dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()
            
            all_classes.append(pred_class)
            all_confidences.append(confidence)
            
            report_lines.append("1. 分类诊断:")
            report_lines.append(f"   预测类别: {CLASS_NAMES[pred_class]}")
            report_lines.append(f"   置信度: {confidence * 100:.2f}%")
            report_lines.append("")
            report_lines.append("   各类别概率分布:")
            for i, class_name in CLASS_NAMES.items():
                prob = probs[i].item()
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                report_lines.append(f"   {class_name:25s}: {prob * 100:6.2f}% {bar}")
            report_lines.append("")
        
        # 分级结果
        if 'grading' in outputs:
            grade_probs = F.softmax(outputs['grading'], dim=1)[0]
            pred_grade = grade_probs.argmax().item()
            grade_confidence = grade_probs[pred_grade].item()
            
            all_grades.append(pred_grade)
            
            report_lines.append("2. 病变分级:")
            report_lines.append(f"   严重程度: {GRADE_NAMES[pred_grade]}")
            report_lines.append(f"   置信度: {grade_confidence * 100:.2f}%")
            report_lines.append("")
            report_lines.append("   各级别概率分布:")
            for i, grade_name in GRADE_NAMES.items():
                prob = grade_probs[i].item()
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                report_lines.append(f"   {grade_name:25s}: {prob * 100:6.2f}% {bar}")
            report_lines.append("")
        
        # 定位结果
        if 'localization' in outputs:
            bbox = outputs['localization'][0].cpu().numpy()
            if bbox.sum() > 0:
                x1, y1, x2, y2 = bbox
                all_bboxes.append(bbox)
                
                report_lines.append("3. 病灶定位:")
                report_lines.append(f"   检测到病灶区域")
                report_lines.append(f"   边界框坐标: ({x1:.3f}, {y1:.3f}) - ({x2:.3f}, {y2:.3f})")
                report_lines.append(f"   区域大小: {(x2 - x1) * (y2 - y1) * 100:.2f}% (相对图像)")
                report_lines.append("")
            else:
                report_lines.append("3. 病灶定位:")
                report_lines.append("   未检测到明显病灶区域")
                report_lines.append("")
        
        # 可解释性
        if 'heatmap' in outputs:
            heatmap = outputs['heatmap'][0].cpu().numpy()
            max_attention = heatmap.max()
            mean_attention = heatmap.mean()
            
            report_lines.append("4. 可解释性分析:")
            report_lines.append(f"   最大注意力值: {max_attention:.4f}")
            report_lines.append(f"   平均注意力值: {mean_attention:.4f}")
            report_lines.append("   热图已生成，请查看可视化图像")
            report_lines.append("")
        
        report_lines.append("-" * 80)
        report_lines.append("")
    
    # 综合诊断
    report_lines.append("【综合诊断意见】")
    report_lines.append("")
    
    if all_classes:
        most_common_class = max(set(all_classes), key=all_classes.count)
        avg_confidence = np.mean(all_confidences)
        
        report_lines.append(f"主要诊断: {CLASS_NAMES[most_common_class]}")
        report_lines.append(f"平均置信度: {avg_confidence * 100:.2f}%")
        report_lines.append("")
    
    if all_grades:
        most_common_grade = max(set(all_grades), key=all_grades.count)
        report_lines.append(f"病变严重程度: {GRADE_NAMES[most_common_grade]}")
        report_lines.append("")
    
    if all_bboxes:
        report_lines.append(f"检测到 {len(all_bboxes)} 个可疑病灶区域")
        report_lines.append("")
    
    # 重要提示
    report_lines.append("=" * 80)
    report_lines.append("重要提示")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("⚠️  本报告由AI模型自动生成，仅供参考，不能替代专业医生的临床诊断。")
    report_lines.append("⚠️  如发现异常，请及时咨询专业医生进行进一步检查。")
    report_lines.append("⚠️  本模型可能未经过充分训练，诊断结果可能存在误差。")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = save_dir / f"{patient_name}_{side}_诊断报告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 同时保存JSON格式
    report_json = {
        "patient_name": patient_name,
        "side": side,
        "check_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "num_images": len(image_paths),
        "diagnoses": []
    }
    
    for idx, (img_path, outputs) in enumerate(zip(image_paths, outputs_list)):
        diagnosis = {
            "image_index": idx + 1,
            "image_path": str(img_path),
            "classification": {},
            "grading": {},
            "localization": {},
            "interpretability": {}
        }
        
        if 'classification' in outputs:
            probs = F.softmax(outputs['classification'], dim=1)[0]
            pred_class = probs.argmax().item()
            diagnosis["classification"] = {
                "predicted_class": pred_class,
                "class_name": CLASS_NAMES[pred_class],
                "confidence": probs[pred_class].item(),
                "all_probabilities": {CLASS_NAMES[i]: probs[i].item() for i in range(4)}
            }
        
        if 'grading' in outputs:
            grade_probs = F.softmax(outputs['grading'], dim=1)[0]
            pred_grade = grade_probs.argmax().item()
            diagnosis["grading"] = {
                "predicted_grade": pred_grade,
                "grade_name": GRADE_NAMES[pred_grade],
                "confidence": grade_probs[pred_grade].item(),
                "all_probabilities": {GRADE_NAMES[i]: grade_probs[i].item() for i in range(3)}
            }
        
        if 'localization' in outputs:
            bbox = outputs['localization'][0].cpu().numpy().tolist()
            diagnosis["localization"] = {
                "bbox": bbox,
                "has_lesion": sum(bbox) > 0
            }
        
        if 'heatmap' in outputs:
            heatmap = outputs['heatmap'][0].cpu().numpy()
            diagnosis["interpretability"] = {
                "max_attention": float(heatmap.max()),
                "mean_attention": float(heatmap.mean()),
                "heatmap_shape": list(heatmap.shape)
            }
        
        report_json["diagnoses"].append(diagnosis)
    
    json_path = save_dir / f"{patient_name}_{side}_诊断报告.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    
    return report_path, json_path


def diagnose_all_patients(mammo_dir, output_dir, model_path=None):
    """
    诊断所有病人的X光片
    
    Args:
        mammo_dir: 钼靶报告文件夹路径
        output_dir: 输出文件夹路径
        model_path: 模型权重路径（可选）
    """
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    print("初始化模型...")
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
    
    # 加载模型权重（如果有）
    if model_path and os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("⚠️  警告: 未找到训练好的模型权重，使用未训练的模型进行推理")
        print("⚠️  诊断结果仅供参考，不可用于实际临床诊断")
    
    model.to(device)
    model.eval()
    
    # 遍历所有病人文件夹
    mammo_dir = Path(mammo_dir)
    patient_dirs = [d for d in mammo_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n找到 {len(patient_dirs)} 个病人文件夹")
    print("=" * 80)
    
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
        
        # 查找所有图像文件
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(patient_dir.glob(ext))
        
        if not image_files:
            print(f"  未找到图像文件，跳过")
            continue
        
        image_files = sorted(image_files)
        print(f"  找到 {len(image_files)} 张图像")
        
        # 处理每张图像
        outputs_list = []
        image_arrays = []
        
        for img_path in image_files:
            # 加载图像
            img_tensor, img_array = load_mammography_image(img_path)
            if img_tensor is None:
                continue
            
            img_tensor = img_tensor.to(device)
            image_arrays.append((img_path, img_array))
            
            # 推理（只使用X光，没有病理图像）
            with torch.no_grad():
                outputs = model(img_tensor, patho_img=None, return_interpretability=True)
                outputs_list.append(outputs)
            
            # 生成可视化
            vis_path = output_dir / f"{patient_name}_{side}_图像{len(outputs_list)}_可视化.png"
            try:
                visualize_diagnosis(img_path, img_array, outputs, vis_path)
            except Exception as e:
                print(f"  生成可视化失败: {e}")
        
        if not outputs_list:
            print(f"  未能处理任何图像，跳过")
            continue
        
        # 生成诊断报告
        image_paths = [path for path, _ in image_arrays]
        report_path, json_path = generate_diagnosis_report(
            patient_name, side, image_paths, outputs_list, output_dir
        )
        
        print(f"  ✓ 诊断完成")
        print(f"    报告: {report_path}")
        print(f"    JSON: {json_path}")
    
    print("\n" + "=" * 80)
    print("✓ 所有病人诊断完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='乳腺X光片诊断')
    parser.add_argument('--mammo_dir', type=str, 
                       default='../钼靶报告',
                       help='钼靶报告文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default='../未命名文件夹 2',
                       help='输出文件夹路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型权重路径（可选）')
    
    args = parser.parse_args()
    
    diagnose_all_patients(args.mammo_dir, args.output_dir, args.model_path)

