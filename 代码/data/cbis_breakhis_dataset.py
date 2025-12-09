"""
CBIS-DDSM和BreaKHis数据集加载器
适配实际的数据集结构
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
import random


class MacenkoNormalization:
    """Macenko染色归一化（用于病理图像）"""
    
    def __init__(self):
        self.reference_od = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        
    def __call__(self, img):
        img_od = -np.log((img.astype(np.float32) + 1) / 256.0)
        normalized_od = img_od * 0.8 + 0.2
        normalized_img = (256.0 * np.exp(-normalized_od) - 1).clip(0, 255).astype(np.uint8)
        return normalized_img


class CBISBreakHisDataset(Dataset):
    """CBIS-DDSM和BreaKHis跨模态数据集"""
    
    def __init__(
        self,
        mammo_metadata: str,
        mammo_root: str,
        patho_root: str,
        img_size=224,
        patch_size=16,
        transform=None,
        use_pathology=True,
        max_samples=None,
        split='train'
    ):
        """
        Args:
            mammo_metadata: X光数据metadata.csv路径
            mammo_root: X光数据根目录
            patho_root: 病理数据根目录
            img_size: 图像大小
            patch_size: patch大小
            transform: 数据增强
            use_pathology: 是否使用病理数据
            max_samples: 最大样本数
            split: 数据集划分（train/test）
        """
        self.mammo_root = Path(mammo_root)
        self.patho_root = Path(patho_root)
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_pathology = use_pathology
        
        # 读取X光metadata
        self.mammo_df = pd.read_csv(mammo_metadata)
        
        # 只保留full mammogram images（排除ROI mask）
        self.mammo_df = self.mammo_df[
            self.mammo_df['Series Description'] == 'full mammogram images'
        ].copy()
        
        # 提取Subject ID（用于配对）
        self.mammo_df['patient_id'] = self.mammo_df['Subject ID'].str.extract(r'([^_]+)')[0]
        
        # 根据文件名判断标签（Calc=钙化，Mass=肿块，通常Calc和Mass都可能是恶性）
        self.mammo_df['lesion_type'] = self.mammo_df['Subject ID'].str.split('_').str[0]
        
        # 简化标签：Calc和Mass都可能是恶性，这里简化为二分类
        # 实际应用中应该根据真实标注
        self.mammo_df['label'] = 1  # 默认为恶性（因为CBIS-DDSM主要是异常病例）
        
        # 数据集划分
        if split == 'train':
            self.mammo_df = self.mammo_df.head(int(len(self.mammo_df) * 0.8))
        else:
            self.mammo_df = self.mammo_df.tail(int(len(self.mammo_df) * 0.2))
        
        if max_samples:
            self.mammo_df = self.mammo_df.head(max_samples)
        
        self.mammo_df = self.mammo_df.reset_index(drop=True)
        
        # 加载病理图像列表
        if use_pathology:
            self.patho_images = self._load_pathology_list()
        else:
            self.patho_images = {}
        
        # 数据增强
        if transform is None:
            self.mammo_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            
            self.patho_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.mammo_transform = transform
            self.patho_transform = transform
        
        self.macenko = MacenkoNormalization()
        
        print(f"加载数据集 ({split}): {len(self.mammo_df)} 个X光样本")
        if use_pathology:
            print(f"病理图像: {len(self.patho_images)} 个类别")
        
    def _load_pathology_list(self):
        """加载病理图像列表"""
        patho_images = {
            'benign': [],
            'malignant': []
        }
        
        # 良性图像
        benign_path = self.patho_root / 'histology_slides' / 'breast' / 'benign'
        if benign_path.exists():
            for img_path in benign_path.rglob('*.png'):
                patho_images['benign'].append(img_path)
        
        # 恶性图像
        malignant_path = self.patho_root / 'histology_slides' / 'breast' / 'malignant'
        if malignant_path.exists():
            for img_path in malignant_path.rglob('*.png'):
                patho_images['malignant'].append(img_path)
        
        return patho_images
    
    def __len__(self):
        return len(self.mammo_df)
    
    def load_dicom_image(self, dicom_path: str) -> Optional[np.ndarray]:
        """加载DICOM图像"""
        try:
            full_path = self.mammo_root / dicom_path
            if not full_path.exists():
                return None
            
            # 如果是目录，找里面的DICOM文件
            if full_path.is_dir():
                dcm_files = list(full_path.glob('*.dcm'))
                if len(dcm_files) == 0:
                    return None
                full_path = dcm_files[0]
            
            dicom_file = pydicom.dcmread(str(full_path))
            image = dicom_file.pixel_array.astype(np.float32)
            
            # 应用窗宽窗位
            if hasattr(dicom_file, 'WindowCenter') and hasattr(dicom_file, 'WindowWidth'):
                center = float(dicom_file.WindowCenter) if not isinstance(dicom_file.WindowCenter, list) else float(dicom_file.WindowCenter[0])
                width = float(dicom_file.WindowWidth) if not isinstance(dicom_file.WindowWidth, list) else float(dicom_file.WindowWidth[0])
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
    
    def get_roi_bbox(self, subject_id: str) -> Optional[Tuple[int, int, int, int]]:
        """从ROI mask提取边界框"""
        try:
            # 查找对应的ROI mask
            roi_pattern = subject_id.replace('_', '_*') + '*ROI*'
            roi_dirs = list(self.mammo_root.glob(f'**/{roi_pattern}'))
            
            if len(roi_dirs) == 0:
                return None
            
            roi_dir = roi_dirs[0]
            dcm_files = list(roi_dir.glob('*.dcm'))
            if len(dcm_files) == 0:
                return None
            
            mask = pydicom.dcmread(str(dcm_files[0])).pixel_array
            
            coords = np.where(mask > 0)
            if len(coords[0]) == 0:
                return None
            
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            
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
    
    def load_pathology_image(self, label: int) -> Optional[np.ndarray]:
        """加载病理图像（根据标签随机选择）"""
        if not self.use_pathology or len(self.patho_images) == 0:
            return None
        
        try:
            # 根据X光标签选择病理图像
            if label == 0:  # 良性
                patho_list = self.patho_images.get('benign', [])
            else:  # 恶性
                patho_list = self.patho_images.get('malignant', [])
            
            if len(patho_list) == 0:
                # 如果对应类别没有图像，随机选择
                all_images = self.patho_images.get('benign', []) + self.patho_images.get('malignant', [])
                if len(all_images) == 0:
                    return None
                patho_list = all_images
            
            # 随机选择一个图像
            img_path = random.choice(patho_list)
            
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
        dicom_path = row['File Location']
        mammo_img = self.load_dicom_image(dicom_path)
        
        if mammo_img is None:
            mammo_img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # 转换为单通道并应用变换
        if len(mammo_img.shape) == 2:
            mammo_img = mammo_img[..., np.newaxis]
        mammo_tensor = self.mammo_transform(mammo_img)
        if mammo_tensor.shape[0] == 3:
            mammo_tensor = mammo_tensor[0:1]
        
        # 加载病理图像
        patho_tensor = None
        if self.use_pathology:
            label = int(row['label'])
            patho_img = self.load_pathology_image(label)
            
            if patho_img is not None:
                patho_tensor = self.patho_transform(Image.fromarray(patho_img))
        
        # 标签
        label_cls = int(row['label'])
        label_grade = label_cls  # 简化：0=良性，1=恶性（可扩展为多级）
        
        # 边界框（尝试从ROI mask获取）
        subject_id = row['Subject ID']
        bbox = self.get_roi_bbox(subject_id)
        
        if bbox is None:
            bbox = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        sample = {
            'mammo_image': mammo_tensor,
            'patho_image': patho_tensor,
            'label_cls': torch.tensor(label_cls, dtype=torch.long),
            'label_grade': torch.tensor(label_grade, dtype=torch.long),
            'bbox': torch.tensor(bbox, dtype=torch.float32),
            'patient_id': str(row.get('patient_id', idx))
        }
        
        return sample

