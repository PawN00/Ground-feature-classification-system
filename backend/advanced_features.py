import numpy as np
import cv2
import io
import base64
import json
from PIL import Image

def generate_confidence_heatmap(conf_array: np.ndarray) -> str:
    """
    将模型预测的概率矩阵转化为热力图 (红=高置信度, 蓝=低置信度)
    :param conf_array: [H, W] 的浮点数组，取值 0~1
    """
    # 转换为 0-255 的灰度图
    conf_img = (conf_array * 255).astype(np.uint8)
    
    # 应用 JET 伪彩映射
    heatmap = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    pil_img = Image.fromarray(heatmap_rgb)
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def mask_to_geojson(mask_np: np.ndarray, class_names: list) -> str:
    """
    将栅格掩码转换为矢量的 GeoJSON 格式，支持导入 GIS 软件
    :param mask_np: [H, W] 的类别索引数组
    :param class_names: 类别名称列表
    """
    features = []
    unique_classes = np.unique(mask_np)
    
    for cls_val in unique_classes:
        if cls_val == 0:  # 通常 0 是背景，如果不需要背景矢量化可跳过
            continue
            
        # 提取当前类的二值掩码
        bin_mask = (mask_np == cls_val).astype(np.uint8) * 255
        
        # 寻找多边形轮廓
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 抽稀多边形边缘，平滑且减小文件体积
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 3:
                coords = approx.reshape(-1, 2).tolist()
                coords.append(coords[0])  # 闭合多边形
                
                features.append({
                    "type": "Feature",
                    "properties": {
                        "class_id": int(cls_val),
                        "class_name": class_names[int(cls_val)] if int(cls_val) < len(class_names) else "Unknown"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                })
                
    geojson_dict = {
        "type": "FeatureCollection",
        "features": features
    }
    return json.dumps(geojson_dict, ensure_ascii=False)