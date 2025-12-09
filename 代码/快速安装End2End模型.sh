#!/bin/bash
# 快速安装End-to-end模型的脚本

echo "=========================================="
echo "End-to-end模型安装脚本"
echo "=========================================="

# 创建虚拟环境（推荐）
echo ""
echo "步骤1: 创建虚拟环境（可选但推荐）"
read -p "是否创建conda虚拟环境? (y/n): " create_env

if [ "$create_env" = "y" ]; then
    echo "创建conda环境: end2end"
    conda create -n end2end python=3.7 -y
    echo "激活环境: conda activate end2end"
    echo "⚠️  请手动激活环境后再运行安装命令"
    echo "   命令: conda activate end2end"
    exit 0
fi

# 安装依赖
echo ""
echo "步骤2: 安装依赖包"
echo "将要安装: keras==2.0.8, tensorflow==1.15.0, h5py, numpy, scipy, pillow, opencv-python"
read -p "是否继续? (y/n): " install_deps

if [ "$install_deps" = "y" ]; then
    pip3 install keras==2.0.8
    pip3 install tensorflow==1.15.0
    pip3 install "numpy<1.20"
    pip3 install h5py scipy pillow opencv-python pandas
    echo "✅ 依赖安装完成"
fi

# 创建权重目录
echo ""
echo "步骤3: 创建权重目录"
mkdir -p ../pretrained_models/end-to-end_all_convolutional_design/weights
echo "✅ 权重目录已创建: ../pretrained_models/end-to-end_all_convolutional_design/weights"

# 下载权重说明
echo ""
echo "步骤4: 下载预训练权重"
echo "=========================================="
echo "请访问以下链接下载权重文件:"
echo "https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc"
echo ""
echo "推荐下载:"
echo "  - INbreast VGG16 [512-512-1024]x2 (AUC 0.95-0.96) ⭐"
echo "  - DDSM Resnet50 [512-512-1024]x2 (AUC 0.86-0.91)"
echo ""
echo "下载后将.h5文件保存到:"
echo "  ../pretrained_models/end-to-end_all_convolutional_design/weights/"
echo "=========================================="

# 测试安装
echo ""
read -p "是否测试Keras和TensorFlow安装? (y/n): " test_install

if [ "$test_install" = "y" ]; then
    python3 << EOF
try:
    import keras
    print(f"✅ Keras版本: {keras.__version__}")
except ImportError:
    print("❌ Keras未安装")

try:
    import tensorflow as tf
    print(f"✅ TensorFlow版本: {tf.__version__}")
except ImportError:
    print("❌ TensorFlow未安装")
EOF
fi

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 下载权重文件到weights目录"
echo "2. 运行诊断脚本:"
echo "   python3 diagnose_with_high_accuracy_model.py \\"
echo "       --mammo_dir \"../钼靶报告\" \\"
echo "       --output_dir \"../未命名文件夹 2\" \\"
echo "       --end2end_model \"../pretrained_models/end-to-end_all_convolutional_design/weights/your_model.h5\""


