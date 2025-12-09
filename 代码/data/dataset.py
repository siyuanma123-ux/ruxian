"""
跨模态乳腺癌数据集加载器
支持X光（DICOM）和病理（PNG）图像
"""

import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2


class MacenkoNormalization:
    """
    Macenko染色归一化（用于病理图像）
    将不同染色的病理图像归一化到统一风格
    """
    
    def __init__(self):
        # 参考染色矩阵（H&E标准）
        self.reference_od = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        
    def __call__(self, img):
        """
        Args:
            img: [H, W, 3] RGB图像
        Returns:
            normalized_img: 归一化后的图像
        """
        # 转换为OD（Optical Density）空间
        img_od = -np.log((img.astype(np.float32) + 1) / 256.0)
        
        # 提取染色向量（简化版）
        # 实际实现需要更复杂的SVD分解
        
        # 归一化到参考染色
        # 这里使用简化版本
        normalized_od = img_od * 0.8 + 0.2  # 简化处理
        
        # 转换回RGB
        normalized_img = (256.0 * np.exp(-normalized_od) - 1).clip(0, 255).astype(np.uint8)
        
        return normalized_img


class CrossModalBreastCancerDataset(Dataset):
    """跨模态乳腺癌数据集"""
    
    def __init__(
        self,
        mammo_csv: str,
        patho_root: str,
        mammo_root: str,
        img_size=224,
        patch_size=16,
        transform=None,
        use_pathology=True,
        max_samples=None
    ):
        """
        Args:
            mammo_csv: X光数据CSV文件路径
            patho_root: 病理图像根目录
            mammo_root: X光图像根目录
            img_size: 图像大小
            patch_size: patch大小
            transform: 数据增强
            use_pathology: 是否使用病理数据
            max_samples: 最大样本数（用于调试）
        """
        self.mammo_root = Path(mammo_root)
        self.patho_root = Path(patho_root)
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_pathology = use_pathology
        
        # 读取X光数据
        self.mammo_df = pd.read_csv(mammo_csv)
        if max_samples:
            self.mammo_df = self.mammo_df.head(max_samples)
        
        # 清理数据
        self.mammo_df = self.mammo_df.dropna(subset=['image file path'])
        
        # 标签映射
        self.label_map = {
            'BENIGN': 0,
            'BENIGN_WITHOUT_CALLBACK': 0,
            'MALIGNANT': 1,
            'MALIGNANT_WITHOUT_CALLBACK': 1
        }
        
        self.grade_map = {
            'BENIGN': 0,
            'BENIGN_WITHOUT_CALLBACK': 0,
            'MALIGNANT': 1,  # In situ
            'MALIGNANT_WITHOUT_CALLBACK': 2  # Invasive
        }
        
        # 数据增强
        if transform is None:
            self.mammo_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
            ])
            
            self.patho_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.mammo_transform = transform
            self.patho_transform = transform
        
        # Macenko归一化
        self.macenko = MacenkoNormalization()
        
        print(f"加载数据集: {len(self.mammo_df)} 个X光样本")
        
    def __len__(self):
        return len(self.mammo_df)
    
    def load_dicom_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """加载DICOM图像"""
        try:
            full_path = self.mammo_root / dicom_path
            if not full_path.exists():
                return None
            
            dicom_file = pydicom.dcmread(str(full_path))
            image = dicom_file.pixel_array.astype(np.float32)
            
            # 应用窗宽窗位
            if hasattr(dicom_file, 'WindowCenter') and hasattr(dicom_file, 'WindowWidth'):
                center = float(dicom_file.WindowCenter)
                width = float(dicom_file.WindowWidth)
                img_min = max(0, center - width // 2)
                img_max = min(image.max(), center + width // 2)
                image = np.clip(image, img_min, img_max)
            
            # 归一化到0-255
            if image.max() > image.min():
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)
            
            # 直方图均衡化
            image = cv2.equalizeHist(image)
            
            return image
        except Exception as e:
            print(f"加载DICOM失败: {dicom_path}, 错误: {e}")
            return None
    
    def get_roi_bbox(self, mask_path: str) -> Optional[Tuple[int, int, int, int]]:
        """从ROI mask提取边界框"""
        try:
            full_path = self.mammo_root / mask_path
            if not full_path.exists():
                return None
            
            mask = pydicom.dcmread(str(full_path)).pixel_array
            
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
            # 归一化到[0, 1]
            h, w = mask.shape
            bbox = np.array([
                x_min / w,
                y_min / h,
                (x_max - x_min) / w,
                (y_max - y_min) / h
            ], dtype=np.float32)
            
            return bbox
        except:
            return None
    
    def load_pathology_image(self, patient_id: str, label: str) -> Optional[np.ndarray]:
        """加载病理图像"""
        if not self.use_pathology:
            return None
        
        try:
            # 根据标签查找病理图像
            if label in ['BENIGN', 'BENIGN_WITHOUT_CALLBACK']:
                path_type = 'benign'
            else:
                path_type = 'malignant'
            
            # 搜索病理图像
            search_paths = [
                self.patho_root / f'breast/{path_type}/**/*.png',
                self.patho_root / f'breast/{path_type}/**/*.jpg'
            ]
            
            images = []
            for pattern in search_paths:
                images.extend(list(self.patho_root.glob(str(pattern))))
            
            if len(images) == 0:
                return None
            
            # 使用patient_id的hash选择图像（保证一致性）
            img_idx = hash(patient_id) % len(images)
            img_path = images[img_idx]
            
            # 加载图像
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
            
            # Macenko归一化
            img = self.macenko(img)
            
            return img
        except Exception as e:
            print(f"加载病理图像失败: {e}")
            return None
    
    def __getitem__(self, idx):
        row = self.mammo_df.iloc[idx]
        
        # 加载X光图像
        mammo_path = row['image file path']
        mammo_img = self.load_dicom_image(mammo_path)
        
        if mammo_img is None:
            # 如果加载失败，返回空样本
            mammo_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 转换为单通道并应用变换
        if len(mammo_img.shape) == 2:
            mammo_img = mammo_img[..., np.newaxis]
        mammo_tensor = self.mammo_transform(mammo_img)
        if mammo_tensor.shape[0] == 3:
            mammo_tensor = mammo_tensor[0:1]  # 只保留第一个通道
        
        # 加载病理图像
        patho_tensor = None
        if self.use_pathology:
            patient_id = str(row.get('patient_id', idx))
            label = str(row.get('pathology', row.get('assessment', 'BENIGN')))
            patho_img = self.load_pathology_image(patient_id, label)
            
            if patho_img is not None:
                patho_tensor = self.patho_transform(Image.fromarray(patho_img))
        
        # 标签
        label_str = str(row.get('pathology', row.get('assessment', 'BENIGN')))
        label_cls = self.label_map.get(label_str, 0)
        label_grade = self.grade_map.get(label_str, 0)
        
        # 边界框
        bbox = None
        if 'ROI mask file path' in row and pd.notna(row['ROI mask file path']):
            bbox = self.get_roi_bbox(row['ROI mask file path'])
        
        if bbox is None:
            bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # 默认全图
        
        sample = {
            'mammo_image': mammo_tensor,
            'patho_image': patho_tensor,
            'label_cls': torch.tensor(label_cls, dtype=torch.long),
            'label_grade': torch.tensor(label_grade, dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'patient_id': patient_id if self.use_pathology else str(idx)
        }
        
        return sample

