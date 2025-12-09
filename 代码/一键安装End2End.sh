#!/bin/bash
# 一键安装End-to-end模型的脚本

echo "=========================================="
echo "End-to-end模型一键安装脚本"
echo "=========================================="

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ conda未安装，请先安装Anaconda或Miniconda"
    exit 1
fi

echo ""
echo "步骤1: 创建conda环境 (Python 3.7)"
read -p "是否创建conda环境 'end2end'? (y/n): " create_env

if [ "$create_env" = "y" ]; then
    conda create -n end2end python=3.7 -y
    echo "✅ 环境创建完成"
    echo ""
    echo "⚠️  请手动激活环境:"
    echo "   conda activate end2end"
    echo ""
    echo "然后运行以下命令安装依赖:"
    echo "   pip install keras==2.0.8 tensorflow==1.15.0 h5py scipy pillow opencv-python pandas \"numpy<1.20\""
    exit 0
fi

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  未检测到conda环境，建议先激活end2end环境"
    read -p "是否继续安装? (y/n): " continue_install
    if [ "$continue_install" != "y" ]; then
        exit 0
    fi
fi

echo ""
echo "步骤2: 安装依赖包"
pip install keras==2.0.8
pip install tensorflow==1.15.0
pip install h5py scipy pillow opencv-python pandas "numpy<1.20"

echo ""
echo "步骤3: 验证安装"
python -c "import keras; import tensorflow as tf; print('✅ Keras:', keras.__version__); print('✅ TensorFlow:', tf.__version__)" 2>&1

echo ""
echo "步骤4: 检查权重目录"
mkdir -p pretrained_models/end-to-end_all_convolutional_design/weights
echo "✅ 权重目录: pretrained_models/end-to-end_all_convolutional_design/weights"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "下一步: 从Google Drive下载权重文件"
echo "链接: https://drive.google.com/drive/folders/0B1PVLadG_dCKV2pZem5MTjc1cHc"
