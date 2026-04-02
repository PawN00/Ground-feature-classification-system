import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import io
import base64
import gc
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from yolo_model import YOLOv11Segmenter
from EfficientFormer_model import EfficientFormerSegmenter
from UNet import UNetPlusPlusSegmenter
MODEL_PATH = r'D:\RuanZhu-wc-v2\backend\best_model.pth'

# 模型参数（和你训练时完全一致）
IN_CHANNELS = 4
NUM_CLASSES = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别映射（完全复用你的dataset里的LABEL_MAPPING）
try:
    from dataset import LABEL_MAPPING
except ImportError:
    LABEL_MAPPING = {
        0: '背景',
        1: '农田',
        2: '建筑',
        3: '森林',
        4: '水体',
        5: '道路'
    }

# 类别颜色映射（和你预测代码完全一致）
CLASS_COLORS = {
    0: (0, 0, 0),        # 背景 - 黑色
    1: (0, 255, 0),      # 农田 - 绿色
    2: (255, 0, 0),      # 建筑 - 红色
    3: (0, 0, 255),      # 森林 - 蓝色
    4: (255, 255, 0),    # 水体 - 黄色
    5: (255, 0, 255),    # 道路 - 紫色
}



def get_infer_transform():
    # 请确保这里的预处理和你训练时的transform完全一致
    return transforms.Compose([
        transforms.Resize((256, 256)),  # 和你训练时的输入尺寸一致
        transforms.ToTensor(),
        # 如果你训练时有Normalize，取消下面注释并修改参数
        # transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])
    ])


model = None

def load_model():
    global model
    print("="*50)
    print(f"正在加载模型到 {DEVICE}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
        return False

    try:
        # 初始化模型（和你训练时完全一致）
        model = YOLOv11Segmenter(
            in_channels=IN_CHANNELS,
            num_classes=NUM_CLASSES
        )
        
        # 加载权重（和你预测代码完全一致）
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval()
        print("✅ 模型加载成功！")
        print("="*50)
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_image_from_bytes(file_bytes):
    if model is None:
        return None, "模型未加载，请检查后端控制台报错"

    try:
        # 1. 读取影像（兼容4通道TIF、多光谱遥感图）
        img_pil = Image.open(io.BytesIO(file_bytes))
        print(f"调试信息：影像模式={img_pil.mode}, 尺寸={img_pil.size}")
        
        # 强制处理为4通道（和你训练时的IN_CHANNELS=4完全匹配）
        if IN_CHANNELS == 4:
            if img_pil.mode == 'RGBA':
                # 已经是4通道，直接使用
                pass
            elif img_pil.mode == 'RGB':
                # 3通道RGB，自动填充第4通道（近红外）为全255
                r, g, b = img_pil.split()
                ir_channel = Image.new('L', img_pil.size, 255)
                img_pil = Image.merge('RGBA', (r, g, b, ir_channel))
            else:
                # 其他模式，强制转为RGBA
                img_pil = img_pil.convert('RGBA')
        elif IN_CHANNELS == 3:
            img_pil = img_pil.convert('RGB')
        
        original_size = img_pil.size  # 保存原始尺寸，用于还原结果

        # 2. 预处理（和训练时完全一致）
        transform = get_infer_transform()
        input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
        print(f"调试信息：输入张量形状={input_tensor.shape}")

        # 3. 模型推理（和你预测代码完全一致）
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            prediction_np = predictions.cpu().numpy()[0]  # [H, W]

        # 4. 生成彩色分割结果图
        h, w = prediction_np.shape
        pred_color = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS.items():
            pred_color[prediction_np == class_id] = color

        # 还原为原始影像尺寸
        result_img = Image.fromarray(pred_color)
        result_img = result_img.resize(original_size, Image.NEAREST)  # 最近邻插值，保证标签锐利

        # 5. 转Base64，用于前端显示
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # 6. 地物占比统计（和你预测代码逻辑一致）
        unique, counts = np.unique(prediction_np, return_counts=True)
        total_pixels = prediction_np.size
        area_stats = {
            LABEL_MAPPING.get(u, f"未知类别{u}"): round((c / total_pixels) * 100, 2)
            for u, c in zip(unique, counts)
        }

        # 7. 生成前端可预览的原图Base64（解决TIF无法预览的核心问题）
        preview_img = img_pil.convert('RGB')  # 转为浏览器支持的RGB格式
        preview_buffered = io.BytesIO()
        preview_img.save(preview_buffered, format="PNG")
        preview_base64 = base64.b64encode(preview_buffered.getvalue()).decode()

        return {
            'success': True,
            'preview_base64': preview_base64,  # 前端可显示的原图
            'result_base64': img_base64,       # 分割结果图
            'stats': area_stats
        }

    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
# ==========================================
# 新增：批量预测函数（支持文件夹多文件）
# ==========================================
def batch_predict_images(file_list, save_root="predict_results"):
    """
    批量预测多张影像
    Args:
        file_list: 列表，每个元素是 {filename: 文件名, file_bytes: 文件字节流}
        save_root: 结果保存根目录
    Returns:
        dict: 包含批量结果、保存路径、第一张图的预览数据
    """
    if model is None:
        return None, "模型未加载，请检查后端"

    # 按时间戳创建保存目录，避免覆盖
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(save_root, timestamp)
    original_save_dir = os.path.join(save_dir, "original_images")
    pred_save_dir = os.path.join(save_dir, "prediction_results")
    os.makedirs(original_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)

    batch_results = []
    first_image_preview = None

    try:
        for idx, file_info in enumerate(tqdm(file_list, desc="批量预测中")):
            filename = file_info["filename"]
            file_bytes = file_info["file_bytes"]

            # 1. 保存原始影像
            original_save_path = os.path.join(original_save_dir, filename)
            with open(original_save_path, "wb") as f:
                f.write(file_bytes)

            # 2. 执行单张预测（复用你已有的单张预测逻辑）
            pred_result = predict_image_from_bytes(file_bytes)
            if not pred_result or not pred_result["success"]:
                batch_results.append({
                    "filename": filename,
                    "success": False,
                    "error": pred_result.get("error", "预测失败")
                })
                continue

            # 3. 保存预测结果图
            pred_img = Image.open(io.BytesIO(base64.b64decode(pred_result["result_base64"])))
            pred_save_path = os.path.join(pred_save_dir, f"{os.path.splitext(filename)[0]}_pred.png")
            pred_img.save(pred_save_path)

            # 4. 保存统计信息
            stats_save_path = os.path.join(pred_save_dir, f"{os.path.splitext(filename)[0]}_stats.txt")
            with open(stats_save_path, "w", encoding="utf-8") as f:
                f.write(f"影像名称: {filename}\n")
                f.write(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*30 + "\n")
                f.write("地物占比统计:\n")
                for class_name, percent in pred_result["stats"].items():
                    f.write(f"  {class_name}: {percent}%\n")

            # 5. 整理结果
            result_item = {
                "filename": filename,
                "success": True,
                "original_save_path": original_save_path,
                "pred_save_path": pred_save_path,
                "stats": pred_result["stats"]
            }
            batch_results.append(result_item)

            # 6. 记录第一张图的预览数据，用于前端展示
            if idx == 0:
                first_image_preview = {
                    "preview_base64": pred_result["preview_base64"],
                    "result_base64": pred_result["result_base64"],
                    "stats": pred_result["stats"],
                    "filename": filename
                }

        # 返回最终结果
        return {
            "success": True,
            "save_dir": os.path.abspath(save_dir),
            "total_count": len(file_list),
            "success_count": len([r for r in batch_results if r["success"]]),
            "fail_count": len([r for r in batch_results if not r["success"]]),
            "batch_results": batch_results,
            "first_image": first_image_preview
        }, None

    except Exception as e:
        print(f"批量预测出错: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)