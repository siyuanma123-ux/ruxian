"""
使用MV-Swin-T模型进行诊断
"""

import os
import sys
import torch
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

# 添加MV-Swin-T路径
mvswint_path = Path(__file__).parent.parent / "MV-Swin-T-main"
if not mvswint_path.exists():
    print(f"❌ 找不到MV-Swin-T-main文件夹: {mvswint_path}")
    sys.exit(1)

# 添加路径 - 需要将MV-Swin-T-main目录添加到路径
# 必须在最前面，这样models模块才能正确导入
sys.path.insert(0, str(mvswint_path))

# 切换到MV-Swin-T目录以便相对导入能工作
original_cwd = os.getcwd()
original_path = sys.path.copy()

try:
    # 切换到MV-Swin-T目录
    os.chdir(str(mvswint_path))
    # 确保当前目录在路径中
    if str(mvswint_path) not in sys.path:
        sys.path.insert(0, str(mvswint_path))
    if '.' not in sys.path:
        sys.path.insert(0, '.')
    
    # 需要先安装timm: pip install timm
    import timm
    
    # 在MV-Swin-T目录下导入
    from models.mvswintransformer import MVSwinTransformer
    
    print("✅ MV-Swin-T模型导入成功")
except ImportError as e:
    print(f"❌ 导入MV-Swin-T模型失败: {e}")
    print("请安装依赖: pip install timm")
    print("或运行: pip3 install timm --user")
    import traceback
    traceback.print_exc()
    os.chdir(original_cwd)
    sys.path = original_path
    sys.exit(1)
except Exception as e:
    print(f"❌ 发生错误: {e}")
    import traceback
    traceback.print_exc()
    os.chdir(original_cwd)
    sys.path = original_path
    sys.exit(1)
finally:
    # 恢复工作目录和路径
    os.chdir(original_cwd)
    # 注意：不恢复sys.path，因为后续代码可能需要models模块

# 类别名称
CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "异常 (Abnormal)"
}


def load_and_preprocess_image(image_path, target_size=384):
    """加载和预处理图像"""
    try:
        # 加载图像
        img = Image.open(image_path).convert('RGB')
        
        # Resize到目标尺寸
        img = img.resize((target_size, target_size), Image.LANCZOS)
        
        # 转换为numpy数组并归一化到[0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # 转换为tensor: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # 添加batch维度: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor, img_array
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None, None


def load_mvswint_model(model_path=None, num_classes=1, target_size=384, window_size=12):
    """加载MV-Swin-T模型"""
    try:
        # 确保使用CPU，避免CUDA相关错误
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用CUDA
        
        # 创建模型
        model = MVSwinTransformer(
            img_size=target_size,
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_layer=torch.nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False
        )
        
        # 如果有预训练权重，加载它
        if model_path and os.path.exists(model_path):
            print(f"加载预训练权重: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("✅ 预训练权重加载成功")
            except Exception as e:
                print(f"⚠️  加载权重失败: {e}")
                print("   使用随机初始化的模型")
        else:
            print("⚠️  未找到预训练权重，使用随机初始化的模型")
            print("   注意: 随机模型的结果不可靠，仅供参考")
        
        model.eval()
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def diagnose_with_mvswint(model, img_tensor, device, threshold=0.5):
    """使用MV-Swin-T模型进行诊断"""
    model.eval()
    
    # MV-Swin-T需要两个视图（CC和MLO）
    # 如果没有配对视图，使用同一张图像作为两个视图
    img_cc = img_tensor.to(device)
    img_mlo = img_tensor.to(device)  # 使用同一张图像
    
    with torch.no_grad():
        # 前向传播
        output = model(img_cc, img_mlo)
        
        # 如果是二分类，应用sigmoid
        if output.shape[1] == 1:
            prob = torch.sigmoid(output)[0, 0].item()
            pred_class = 1 if prob > threshold else 0
        else:
            # 多分类
            probs = F.softmax(output, dim=1)[0]
            prob = probs[1].item()  # 异常概率
            pred_class = probs.argmax().item()
    
    return {
        'pred_class': pred_class,
        'confidence': prob,
        'probabilities': {
            '正常': 1.0 - prob,
            '异常': prob
        }
    }


def visualize_diagnosis_with_annotation(image_path, image_array, result, save_path):
    """可视化诊断结果，在图像上标注"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # 显示原始图像
    if len(image_array.shape) == 3:
        # RGB图像
        ax.imshow(image_array)
    else:
        # 灰度图像
        ax.imshow(image_array, cmap='gray')
    ax.axis('off')
    
    # 添加诊断标注
    pred_class = result['pred_class']
    confidence = result['confidence']
    class_name = CLASS_NAMES[pred_class]
    
    # 获取图像尺寸用于像素坐标标注
    h, w = image_array.shape[:2]
    
    # 在图像左上角添加诊断信息框
    info_box_y = h * 0.05
    info_box_x = w * 0.02
    
    # 诊断结果（大号，醒目）
    result_color = 'red' if pred_class == 1 else 'green'
    result_bg = 'yellow' if pred_class == 1 else 'lightgreen'
    
    ax.text(info_box_x, info_box_y, f'诊断结果: {class_name}', 
            fontsize=20, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=result_bg, alpha=0.9, edgecolor=result_color, linewidth=3),
            verticalalignment='top', family='sans-serif')
    
    # 置信度
    ax.text(info_box_x, info_box_y + h * 0.08, f'置信度: {confidence * 100:.2f}%',
            fontsize=16, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='blue', linewidth=2),
            verticalalignment='top', family='sans-serif')
    
    # 概率分布
    prob_text = f"正常: {(1-confidence)*100:.1f}%  |  异常: {confidence*100:.1f}%"
    ax.text(info_box_x, info_box_y + h * 0.16, prob_text,
            fontsize=14, color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=1),
            verticalalignment='top', family='sans-serif')
    
    # 在图像中心添加大号标注（如果异常）
    center_x, center_y = w / 2, h / 2
    if pred_class == 1:
        # 异常：红色警告
        ax.text(center_x, center_y, '⚠️\n异常', 
                fontsize=min(w, h) // 8, fontweight='bold',
                color='red',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=4),
                family='sans-serif')
    else:
        # 正常：绿色对勾
        ax.text(center_x, center_y, '✓\n正常', 
                fontsize=min(w, h) // 8, fontweight='bold',
                color='green',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=4),
                family='sans-serif')
    
    # 在图像四个角添加边框（红色表示异常，绿色表示正常）
    border_color = 'red' if pred_class == 1 else 'green'
    border_width = 8
    
    # 上边框
    ax.plot([0, w], [0, 0], color=border_color, linewidth=border_width)
    # 下边框
    ax.plot([0, w], [h, h], color=border_color, linewidth=border_width)
    # 左边框
    ax.plot([0, 0], [0, h], color=border_color, linewidth=border_width)
    # 右边框
    ax.plot([w, w], [0, h], color=border_color, linewidth=border_width)
    
    # 不显示模型信息
    
    # 添加时间戳（左下角）
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    ax.text(w * 0.02, h * 0.98, timestamp,
            fontsize=10, color='gray',
            ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            family='sans-serif')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0)
    plt.close()


def generate_diagnosis_report(patient_name, side, image_paths, results_list, save_dir):
    """生成诊断报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("乳腺X光诊断报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"病人姓名: {patient_name}")
    report_lines.append(f"检查部位: {side}")
    report_lines.append(f"检查日期: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append(f"图像数量: {len(image_paths)}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for idx, (img_path, result) in enumerate(zip(image_paths, results_list)):
        if result is None:
            continue
        
        report_lines.append(f"【图像 {idx + 1}】")
        report_lines.append(f"文件路径: {img_path}")
        report_lines.append("")
        report_lines.append("诊断结果:")
        report_lines.append(f"  预测类别: {CLASS_NAMES[result['pred_class']]}")
        report_lines.append(f"  置信度: {result['confidence'] * 100:.2f}%")
        report_lines.append("")
        report_lines.append("概率分布:")
        for class_name, prob in result['probabilities'].items():
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report_lines.append(f"  {class_name:25s}: {prob * 100:6.2f}% {bar}")
        report_lines.append("")
        report_lines.append("-" * 80)
        report_lines.append("")
    
    # 综合诊断
    report_lines.append("【综合诊断意见】")
    report_lines.append("")
    
    all_predictions = [r['pred_class'] for r in results_list if r is not None]
    if all_predictions:
        most_common = max(set(all_predictions), key=all_predictions.count)
        agreement = all_predictions.count(most_common) / len(all_predictions)
        
        report_lines.append(f"模型一致性: {agreement * 100:.1f}%")
        report_lines.append(f"主要预测: {CLASS_NAMES[most_common]}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = save_dir / f"{patient_name}_{side}_诊断报告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # 保存JSON
    json_data = {
        'patient_name': patient_name,
        'side': side,
        'check_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_images': len(image_paths),
        'diagnoses': []
    }
    
    for idx, (img_path, result) in enumerate(zip(image_paths, results_list)):
        if result:
            json_data['diagnoses'].append({
                'image_index': idx + 1,
                'image_path': str(img_path),
                'predicted_class': result['pred_class'],
                'class_name': CLASS_NAMES[result['pred_class']],
                'confidence': result['confidence'],
                'probabilities': result['probabilities']
            })
    
    json_path = save_dir / f"{patient_name}_{side}_诊断报告.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    return report_path, json_path


def diagnose_all_patients(mammo_dir, output_dir, model_path=None):
    """诊断所有病人"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载MV-Swin-T模型...")
    model = load_mvswint_model(model_path, num_classes=1, target_size=384, window_size=12)
    if model is None:
        print("❌ 模型加载失败，退出")
        return
    
    model.to(device)
    print("✅ 模型加载完成")
    
    # 遍历所有病人
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
        results_list = []
        image_paths = []
        
        for img_path in image_files:
            # 加载和预处理图像
            img_tensor, img_array = load_and_preprocess_image(img_path, target_size=384)
            if img_tensor is None:
                continue
            
            image_paths.append(img_path)
            
            # 诊断
            result = diagnose_with_mvswint(model, img_tensor, device, threshold=0.5)
            results_list.append(result)
            
            # 生成带标注的可视化图像
            vis_path = output_dir / f"{patient_name}_{side}_图像{len(results_list)}_标注.png"
            try:
                # 将图像数组转换回0-255范围用于显示
                img_display = (img_array * 255).astype(np.uint8)
                visualize_diagnosis_with_annotation(img_path, img_display, result, vis_path)
            except Exception as e:
                print(f"  生成可视化失败: {e}")
        
        if not results_list:
            print(f"  未能处理任何图像，跳过")
            continue
        
        # 生成诊断报告
        report_path, json_path = generate_diagnosis_report(
            patient_name, side, image_paths, results_list, output_dir
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
    
    parser = argparse.ArgumentParser(description='使用MV-Swin-T模型进行诊断')
    parser.add_argument('--mammo_dir', type=str, 
                       default='../钼靶报告',
                       help='钼靶报告文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default='../未命名文件夹',
                       help='输出文件夹路径')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型权重路径（可选）')
    
    args = parser.parse_args()
    
    diagnose_all_patients(args.mammo_dir, args.output_dir, args.model_path)

