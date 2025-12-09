"""
使用高准确率模型进行诊断
支持多种模型：End-to-end All Conv模型（如果有权重）或使用PyTorch模型
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

BINARY_CLASS_NAMES = {
    0: "正常 (Normal)",
    1: "异常 (Abnormal)"
}


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


def try_load_end2end_model(model_path=None):
    """尝试加载End-to-end All Conv模型"""
    try:
        import keras
        print(f"检测到Keras版本: {keras.__version__}")
        
        # 检查模型路径
        if model_path is None:
            # 查找默认路径
            default_paths = [
                '../pretrained_models/end-to-end_all_convolutional_design/weights/inbreast_vgg16_[512-512-1024]x2.h5',
                '../pretrained_models/end-to-end_all_convolutional_design/weights/ddsm_resnet50_[512-512-1024]x2.h5',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path is None or not os.path.exists(model_path):
            print("⚠️  未找到End-to-end模型的预训练权重")
            print("   请从Google Drive下载: https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc")
            return None
        
        print(f"加载模型: {model_path}")
        model = keras.models.load_model(model_path)
        print("✅ End-to-end模型加载成功")
        return model
    except ImportError:
        print("⚠️  Keras未安装，无法使用End-to-end模型")
        print("   安装命令: pip install keras==2.0.8 tensorflow==1.15.0")
        return None
    except Exception as e:
        print(f"⚠️  加载End-to-end模型失败: {e}")
        return None


def diagnose_with_end2end_model(model, img_tensor, img_array):
    """使用End-to-end模型进行诊断"""
    try:
        import keras
        import numpy as np
        
        # 转换图像格式
        # End-to-end模型需要1152x896的输入
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_resized = img_pil.resize((1152, 896), Image.LANCZOS)
        img_np = np.array(img_resized, dtype=np.float32)
        
        # 预处理（根据模型要求）
        # rescale_factor for PNG: 0.003891
        img_np = img_np * 0.003891
        img_np = (img_np - 44.33) / 255.0  # featurewise mean normalization
        
        # 调整维度: (H, W) -> (1, H, W, 1)
        img_np = np.expand_dims(np.expand_dims(img_np, axis=0), axis=-1)
        
        # 预测
        pred = model.predict(img_np, batch_size=1, verbose=0)
        
        # 二分类输出
        if pred.shape[1] == 2:
            normal_prob = pred[0][0]
            abnormal_prob = pred[0][1]
        else:
            # 单输出，假设是异常概率
            abnormal_prob = float(pred[0][0])
            normal_prob = 1.0 - abnormal_prob
        
        return {
            'model_name': 'End-to-end All Convolutional Design',
            'pred_class': 1 if abnormal_prob > normal_prob else 0,
            'confidence': max(abnormal_prob, normal_prob),
            'probabilities': {
                '正常': normal_prob,
                '异常': abnormal_prob
            },
            'binary': True
        }
    except Exception as e:
        print(f"End-to-end模型推理失败: {e}")
        return None


def diagnose_with_pytorch_model(model, img_tensor, device):
    """使用PyTorch模型进行诊断"""
    model.eval()
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor, patho_img=None, return_interpretability=False)
        logits = outputs['classification']
    
    probs = F.softmax(logits, dim=1)[0]
    pred_class = probs.argmax().item()
    confidence = probs[pred_class].item()
    
    return {
        'model_name': '我们的跨模态模型',
        'pred_class': pred_class,
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: probs[i].item() for i in range(4)
        },
        'binary': False
    }


def generate_diagnosis_report(patient_name, side, image_paths, results_list, save_dir):
    """生成诊断报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("高准确率模型诊断报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"病人姓名: {patient_name}")
    report_lines.append(f"检查部位: {side}")
    report_lines.append(f"检查日期: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    report_lines.append(f"图像数量: {len(image_paths)}")
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("")
    
    for idx, (img_path, results) in enumerate(zip(image_paths, results_list)):
        report_lines.append(f"【图像 {idx + 1}】")
        report_lines.append(f"文件路径: {img_path}")
        report_lines.append("")
        
        for result in results:
            if result is None:
                continue
            
            report_lines.append(f"模型: {result['model_name']}")
            report_lines.append(f"预测类别: {BINARY_CLASS_NAMES.get(result['pred_class'], CLASS_NAMES.get(result['pred_class'], '未知'))}")
            report_lines.append(f"置信度: {result['confidence'] * 100:.2f}%")
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
    
    # 统计所有模型的预测
    all_predictions = []
    for results in results_list:
        for result in results:
            if result:
                all_predictions.append(result['pred_class'])
    
    if all_predictions:
        most_common = max(set(all_predictions), key=all_predictions.count)
        agreement = all_predictions.count(most_common) / len(all_predictions)
        
        report_lines.append(f"模型一致性: {agreement * 100:.1f}%")
        report_lines.append(f"主要预测: {BINARY_CLASS_NAMES.get(most_common, CLASS_NAMES.get(most_common, '未知'))}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("重要提示")
    report_lines.append("=" * 80)
    report_lines.append("⚠️  本报告由AI模型自动生成，仅供参考，不能替代专业医生的临床诊断。")
    report_lines.append("⚠️  如发现异常，请及时咨询专业医生进行进一步检查。")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = save_dir / f"{patient_name}_{side}_高准确率模型诊断报告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_path


def diagnose_all_patients(mammo_dir, output_dir, end2end_model_path=None):
    """诊断所有病人"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 尝试加载End-to-end模型
    print("\n尝试加载高准确率模型...")
    end2end_model = try_load_end2end_model(end2end_model_path)
    
    # 加载我们的PyTorch模型作为备选
    print("\n加载PyTorch模型...")
    pytorch_model = BreastCancerDiagnosisModel(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=4,
        num_grades=3,
        use_dg=True,
        use_interpretability=False
    ).to(device)
    pytorch_model.eval()
    print("✅ PyTorch模型加载完成")
    
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
            # 加载图像
            img_tensor, img_array = load_mammography_image(img_path)
            if img_tensor is None:
                continue
            
            image_paths.append(img_path)
            image_results = []
            
            # 使用End-to-end模型（如果有）
            if end2end_model:
                result = diagnose_with_end2end_model(end2end_model, img_tensor, img_array)
                if result:
                    image_results.append(result)
            
            # 使用PyTorch模型
            result = diagnose_with_pytorch_model(pytorch_model, img_tensor, device)
            image_results.append(result)
            
            results_list.append(image_results)
        
        if not results_list:
            print(f"  未能处理任何图像，跳过")
            continue
        
        # 生成诊断报告
        report_path = generate_diagnosis_report(
            patient_name, side, image_paths, results_list, output_dir
        )
        
        print(f"  ✓ 诊断完成")
        print(f"    报告: {report_path}")
    
    print("\n" + "=" * 80)
    print("✓ 所有病人诊断完成！")
    print(f"结果保存在: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='使用高准确率模型进行诊断')
    parser.add_argument('--mammo_dir', type=str, 
                       default='../钼靶报告',
                       help='钼靶报告文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default='../未命名文件夹 2',
                       help='输出文件夹路径')
    parser.add_argument('--end2end_model', type=str, default=None,
                       help='End-to-end模型权重路径（可选）')
    
    args = parser.parse_args()
    
    diagnose_all_patients(args.mammo_dir, args.output_dir, args.end2end_model)


