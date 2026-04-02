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
import gc
from login import auth_router, oauth2_scheme
from InferenceEngine import InferenceEngine


# ======= 导入拆分出去的登录及用户管理模块 =======
from login import auth_router, oauth2_scheme

app = FastAPI(title="Remote Sensing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 将 login.py 中的所有路由挂载到当前主应用
app.include_router(auth_router)

engine = InferenceEngine()

def img_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/health")
async def health_check():
    return {"model_loaded": engine.model_loaded}

@app.post("/model/switch")
async def switch_model(model_name: str = Form(...), token: str = Depends(oauth2_scheme)):
    model_mapping = {
        "unet++": {
            "build_name": "unet++", 
            "path": r"D:\GID5\U-Net++Shuju\UNetPlusPlus_light_20260324_195942_bs32\best_model.pth"
        },
        "efficientformer": {
            "build_name": "efficientformer", 
            "path": r"D:\GID5\efficientformer\Exp_20260325_210517_bs32\best_model.pth"
        },
        "yolov11": {
            "build_name": "yolov11", 
            "path": r"D:\RuanZhu-wc-v2\backend\best_model.pth"
        }
    }
    if model_name not in model_mapping:
        return {"success": False, "message": "所选模型不存在"}
    
    config = model_mapping[model_name]
    success = engine.load_model(build_name=config["build_name"], model_path=config["path"])
    if success:
        return {"success": True, "message": f"已成功切换至 {model_name} 模型"}
    return {"success": False, "message": "模型加载失败，请检查路径或权重格式"}

# ====== 修改 /predict/single 接口的返回值 ======
@app.post("/predict/single")
async def predict_single(file: UploadFile = File(...), token: str = Depends(oauth2_scheme)):
    if not engine.model_loaded:
        return {"success": False, "message": "请先在右上角切换并加载模型！"}
    try:
        image_bytes = await file.read()
        # 接收5个返回值
        orig_img, pred_pil, stats, heatmap_b64, geojson_str = engine.process_image(image_bytes)
        return {
            "success": True,
            "preview_base64": img_to_base64(orig_img),
            "result_base64": img_to_base64(pred_pil),
            "heatmap_base64": heatmap_b64,
            "geojson": geojson_str,
            "stats": stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"推理失败: {str(e)}"}

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...), token: str = Depends(oauth2_scheme)):
    if not engine.model_loaded:
        return {"success": False, "message": "请先切换或加载模型！"}
        
    save_dir = r"./batch_results"
    os.makedirs(save_dir, exist_ok=True)
    
    success_count, fail_count = 0, 0
    first_image_data = None
    
    for idx, file in enumerate(files):
        try:
            image_bytes = await file.read()
            orig_img, pred_pil, stats = engine.process_image(image_bytes)
            base_name = os.path.splitext(file.filename)[0]
            pred_pil.save(os.path.join(save_dir, f"{base_name}_mask.png"))
            success_count += 1
            if idx == 0:
                first_image_data = {
                    "filename": file.filename, "preview_base64": img_to_base64(orig_img),
                    "result_base64": img_to_base64(pred_pil), "stats": stats
                }
        except Exception as e:
            print(f"文件 {file.filename} 预测失败: {e}")
            fail_count += 1
            
    return {
        "success": True, "total_count": len(files), "success_count": success_count,
        "fail_count": fail_count, "save_dir": os.path.abspath(save_dir), "first_image": first_image_data
    }

if __name__ == "__main__":
    import uvicorn
    try:
        engine.load_model(
            build_name="unet++", 
            model_path=r"D:\GID5\U-Net++Shuju\UNetPlusPlus_light_20260324_195942_bs32\best_model.pth"
        )
    except Exception:
        pass
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)