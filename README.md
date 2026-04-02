没问题，我们把那些“商业级”、“指挥舱”之类的营销词汇和过多的表情符号去掉，回归到一个**纯粹、严谨、专业的开源项目/学术项目**的技术文档风格。

这样的 `README.md` 会显得更加低调务实，适合放在 GitHub 或作为毕业设计/科研项目的说明文档。

-----

# 遥感影像地物分割系统 (Remote Sensing Segmentation System)

本项目是一个基于深度学习的遥感影像地物分割 Web 应用。系统支持多光谱（RGB / 4通道）影像的读取与解析，集成了多种图像分割网络，并提供透明度叠加、置信度热力图、面积估算以及 GeoJSON 矢量导出等功能。

## 核心功能

  * **多模型支持**：后端采用解耦设计，目前已集成 `UNet++ (轻量版)`、`EfficientFormer` 和 `YOLOv11`，支持在前端动态切换推理权重。
  * **图像可视化分析**：
      * 支持预测掩码（Mask）与原始光学影像按自定义透明度叠加显示。
      * 支持提取模型 Softmax 概率，生成置信度伪彩热力图，用于分析模型预测的不确定性。
  * **地理数据处理**：
      * 支持输入影像的空间分辨率 (GSD)，自动换算并统计各类别地物的实际物理面积。
      * 基于 OpenCV 轮廓提取与抽稀算法，支持将栅格掩码转换为 `GeoJSON` 矢量格式导出。
  * **影像兼容性**：后端内置数据预处理模块，兼容 3 通道与 4 通道（含近红外）的 TIFF/PNG/JPG 影像。
  * **基础权限管理**：基于本地 JSON 文件实现轻量级的 RBAC 权限控制，包含 `admin` 与 `user` 角色分离。

-----

## 技术栈

  * **前端**: Vue 3 (Composition API) + Vite + Element Plus + ECharts
  * **后端**: FastAPI + Uvicorn
  * **算法与图像**: PyTorch + OpenCV + Pillow + NumPy

-----

## 项目结构

```text
Project_Root/
├── frontend/                   # Vue 3 前端工程
│   ├── src/
│   │   ├── app.vue             # 核心视图与交互逻辑
│   │   └── main.js             # 前端入口
│   ├── package.json
│   └── vite.config.js
├── backend/                    # FastAPI 后端工程
│   ├── main.py                 # 核心接口与推理逻辑
│   ├── login.py                # 路由拆分：鉴权与用户管理
│   ├── advanced_features.py    # 图像处理扩展（热力图、GeoJSON转换）
│   ├── users.json              # 本地用户数据
│   ├── yolo_model.py           # YOLOv11 网络定义
│   ├── EfficientFormer_model.py# EfficientFormer 网络定义
│   └── UNet.py                 # UNet++ 网络定义
└── README.md                   # 项目说明文档
```

-----

## 部署与运行

### 1\. 后端环境配置

推荐使用 Python 3.8+ 及支持 CUDA 的 GPU 环境。

```bash
# 1. 进入后端目录
cd backend

# 2. 安装依赖
pip install fastapi uvicorn pydantic python-multipart
pip install torch torchvision opencv-python numpy pillow

# 3. 启动服务 (默认运行在 8000 端口)
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2\. 前端环境配置

需提前安装 Node.js (推荐 v16+)。

```bash
# 1. 进入前端目录
cd frontend

# 2. 安装依赖
npm install

# 3. 启动开发服务器
npm run dev
```

打开浏览器访问控制台输出的地址（通常为 `http://localhost:5173`）。

-----

## 使用说明

1.  **账号登录**：系统初始化后，默认管理员账号为 `admin`，密码为 `123456`。管理员具有“账号管理”权限。
2.  **影像上传**：支持单张影像或文件夹批量处理模式。
3.  **参数配置**：
      * 在顶部导航栏选择所需加载的深度学习模型。
      * 在输入框中设置当前影像的空间分辨率（米/像素），用于面积换算。
4.  **解译与导出**：点击执行分割后，可通过工作台右上角的控件调整叠底透明度或切换热力图。右下角提供掩码图（PNG）与矢量数据（GeoJSON）的导出接口。

-----

## 类别映射字典 (Label Mapping)

| Class ID | 类别名称 | 颜色 RGB |
| :--- | :--- | :--- |
| 0 | 背景 (Background) | (0, 0, 0) |
| 1 | 农田 (Farmland) | (0, 255, 0) |
| 2 | 建筑 (Built-up) | (255, 0, 0) |
| 3 | 森林 (Forest) | (0, 0, 255) |
| 4 | 水体 (Water) | (255, 255, 0) |
| 5 | 道路 (Road) | (255, 0, 255) |
