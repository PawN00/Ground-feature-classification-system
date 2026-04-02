import os
import io
import base64
import torch
import numpy as np
from PIL import Image
from typing import List
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import gc
import cv2
from advanced_features import generate_confidence_heatmap, mask_to_geojson
from yolo_model import YOLOv11Segmenter
from EfficientFormer_model import EfficientFormerV2_Seg
from UNet import create_unetplusplus
class InferenceEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_loaded = False
        self.current_model_type = ""
        self.model = None
        
        # 预处理 (与你训练时完全一致)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # 类别颜色映射（完全复刻你的代码）
        self.class_to_color = {
            0: (0, 0, 0),        # 背景 - 黑色
            1: (0, 255, 0),      # 农田 - 绿色
            2: (255, 0, 0),      # 建筑 - 红色
            3: (0, 0, 255),      # 森林 - 蓝色
            4: (255, 255, 0),    # 水体 - 黄色
            5: (255, 0, 255)     # 道路 - 紫色
        }
        self.class_names = ["背景", "农田", "建筑", "森林", "水体", "道路"]

    def decode_segmap(self, prediction_np):
        """将模型输出的类别索引映射为彩色 RGB 图像"""
        h, w = prediction_np.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls_idx, color in self.class_to_color.items():
            rgb[prediction_np == cls_idx] = color
        return rgb

    def load_model(self, build_name: str, model_path: str):
        """动态加载模型：支持三个模型的自由切换"""
        print("="*50)
        print(f"[Info] 正在构建 {build_name} 模型，加载权重: {model_path} to {self.device}")
        self.current_model_type = build_name
        
        if not os.path.exists(model_path):
            print(f"❌ 错误：找不到模型文件 {model_path}")
            return False

        try:
            # 1. 动态实例化对应的网络
            if build_name == "unet++":
                self.model = create_unetplusplus(model_type='light', in_channels=4, num_classes=6).to(self.device)
            elif build_name == "efficientformer":
                self.model = EfficientFormerV2_Seg(in_channels=4, num_classes=6).to(self.device)
            elif build_name == "yolov11":
                self.model = YOLOv11Segmenter(in_channels=4, num_classes=6).to(self.device)
            else:
                raise ValueError(f"未知的模型类型: {build_name}")
            
            # 2. 加载权重
            state_dict = torch.load(model_path, map_location=self.device)
            # 兼容带有 'model_state_dict' 嵌套的权重字典
            if 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                self.model.load_state_dict(state_dict)
                
            self.model.eval()
            self.model_loaded = True
            print("✅ 模型加载成功！")
            print("="*50)
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_image(self, image_bytes: bytes):
        try:
            # 1. 读取影像并处理4通道
            img_pil = Image.open(io.BytesIO(image_bytes))
            original_size = img_pil.size  # (W, H)
            
            if img_pil.mode == 'RGBA':
                pass
            elif img_pil.mode == 'RGB':
                r, g, b = img_pil.split()
                ir_channel = Image.new('L', img_pil.size, 255)
                img_pil = Image.merge('RGBA', (r, g, b, ir_channel))
            else:
                img_pil = img_pil.convert('RGBA')

            preview_img = img_pil.convert('RGB')
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # 2. 模型推理 (新增置信度概率提取)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                max_probs, predictions = torch.max(probs, dim=1)
                
                prediction_np = predictions.cpu().numpy()[0]  # [H_512, W_512]
                conf_np = max_probs.cpu().numpy()[0]          # [H_512, W_512]
                
            # 3. 还原到原始高分辨率 (使用 OpenCV 提速)
            pred_resized = cv2.resize(prediction_np.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
            conf_resized = cv2.resize(conf_np, original_size, interpolation=cv2.INTER_LINEAR)
            
            # 4. 生成彩色分割图
            pred_color = self.decode_segmap(pred_resized)
            result_img = Image.fromarray(pred_color)
            
            # 5. 高级功能：生成热力图与矢量 GeoJSON
            heatmap_base64 = generate_confidence_heatmap(conf_resized)
            geojson_str = mask_to_geojson(pred_resized, self.class_names)
            
            # 6. 统计占比与像素数 (将像素量传给前端，用于计算真实物理面积)
            unique, counts = np.unique(pred_resized, return_counts=True)
            total_pixels = pred_resized.size
            stats = {
                self.class_names[u]: {
                    "percent": round((c / total_pixels) * 100, 2),
                    "pixels": int(c)
                } for u, c in zip(unique, counts)
            }
                
            return preview_img, result_img, stats, heatmap_base64, geojson_str
            
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()