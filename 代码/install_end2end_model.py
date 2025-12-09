"""
安装End-to-end All Convolutional Design模型
包括下载权重文件和安装依赖
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    import sys
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("⚠️  警告: 建议使用Python 3.6-3.8")
    return version

def install_dependencies():
    """安装依赖"""
    print("\n" + "=" * 80)
    print("步骤1: 安装依赖包")
    print("=" * 80)
    
    dependencies = [
        "keras==2.0.8",
        "tensorflow==1.15.0",  # 或 tensorflow-gpu==1.15.0
        "h5py",
        "numpy<1.20",  # 兼容TensorFlow 1.x
        "scipy",
        "pillow",
        "opencv-python",
        "pandas"
    ]
    
    print("\n将要安装的包:")
    for dep in dependencies:
        print(f"  - {dep}")
    
    print("\n⚠️  注意: 这些是较旧的版本，建议在虚拟环境中安装")
    print("   创建虚拟环境: conda create -n end2end python=3.7")
    print("   激活环境: conda activate end2end")
    
    response = input("\n是否继续安装? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消安装")
        return False
    
    print("\n开始安装...")
    for dep in dependencies:
        try:
            print(f"安装 {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dep} 安装失败: {e}")
            return False
    
    print("\n✅ 所有依赖安装完成!")
    return True

def download_weights():
    """下载预训练权重"""
    print("\n" + "=" * 80)
    print("步骤2: 下载预训练权重")
    print("=" * 80)
    
    weights_dir = Path("../pretrained_models/end-to-end_all_convolutional_design/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n预训练权重需要从Google Drive手动下载:")
    print("链接: https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc")
    print("\n推荐下载的模型:")
    print("  1. INbreast VGG16 [512-512-1024]x2 (AUC 0.95-0.96) ⭐ 推荐")
    print("  2. DDSM Resnet50 [512-512-1024]x2 (AUC 0.86-0.91)")
    print("\n下载后请将.h5文件保存到:")
    print(f"  {weights_dir.absolute()}")
    
    # 检查是否已有权重文件
    existing_weights = list(weights_dir.glob("*.h5"))
    if existing_weights:
        print(f"\n✅ 发现已存在的权重文件:")
        for w in existing_weights:
            print(f"  - {w.name}")
        return True
    
    print("\n⚠️  未找到权重文件，请手动下载")
    print("\n尝试使用gdown下载（如果知道文件ID）...")
    
    # 尝试使用gdown（需要文件ID）
    # 注意：这些文件ID需要从Google Drive链接中提取
    file_ids = {
        "inbreast_vgg16": None,  # 需要从Google Drive获取
        "ddsm_resnet50": None
    }
    
    print("由于Google Drive链接需要手动访问，请:")
    print("1. 访问上面的链接")
    print("2. 下载所需的.h5文件")
    print("3. 将文件放到weights目录")
    
    return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 80)
    print("步骤3: 测试模型加载")
    print("=" * 80)
    
    try:
        import keras
        print(f"✅ Keras已安装，版本: {keras.__version__}")
    except ImportError:
        print("❌ Keras未安装")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow已安装，版本: {tf.__version__}")
    except ImportError:
        print("❌ TensorFlow未安装")
        return False
    
    # 检查权重文件
    weights_dir = Path("../pretrained_models/end-to-end_all_convolutional_design/weights")
    weights_files = list(weights_dir.glob("*.h5"))
    
    if not weights_files:
        print("⚠️  未找到权重文件，无法测试模型加载")
        return False
    
    print(f"\n找到 {len(weights_files)} 个权重文件")
    
    # 尝试加载第一个模型
    try:
        model_path = weights_files[0]
        print(f"\n尝试加载模型: {model_path.name}")
        model = keras.models.load_model(str(model_path))
        print(f"✅ 模型加载成功!")
        print(f"   模型输入形状: {model.input_shape}")
        print(f"   模型输出形状: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("End-to-end All Convolutional Design 模型安装脚本")
    print("=" * 80)
    
    # 检查Python版本
    check_python_version()
    
    # 安装依赖
    install_deps = input("\n是否安装依赖? (y/n): ").strip().lower() == 'y'
    if install_deps:
        if not install_dependencies():
            print("\n❌ 依赖安装失败")
            return
    
    # 下载权重
    download_weights()
    
    # 测试模型
    test_model = input("\n是否测试模型加载? (y/n): ").strip().lower() == 'y'
    if test_model:
        test_model_loading()
    
    print("\n" + "=" * 80)
    print("安装完成!")
    print("=" * 80)
    print("\n下一步:")
    print("1. 如果权重文件已下载，可以运行诊断脚本:")
    print("   python3 diagnose_with_high_accuracy_model.py \\")
    print("       --mammo_dir \"../钼靶报告\" \\")
    print("       --output_dir \"../未命名文件夹 2\" \\")
    print("       --end2end_model \"../pretrained_models/end-to-end_all_convolutional_design/weights/your_model.h5\"")
    print("\n2. 如果遇到问题，请查看:")
    print("   - 代码/安装高准确率模型指南.md")
    print("   - 代码/高准确率模型使用指南.md")

if __name__ == '__main__':
    main()


