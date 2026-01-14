# SDXL Inpainting & Multi-Round Editor (Local)

本项目是一个本地 SDXL 应用，结合了 **React 前端**、**FastAPI 后端** 和 **本地 LLM (Qwen)**，支持：
文生图 (古诗理解) → 标注 mask → 多轮局部编辑 → 历史回退。

## ✨ Features

- **Prompt Optimization**: 内置 LLM (Qwen) 理解古诗词，自动生成结构化 prompt (Subject/Action/Composition/Mood)。
- **Generate + Inpaint**: 同一套 API 支持 `text2img` 与 `inpaint`。
- **Modern UI**: 基于 React + Vite 的现代化前端，支持图层蒙版编辑。
- **Mask Processing**: grow / feather / invert + alpha blend 后融合。
- **Session & History**: 多轮版本、回退、缩略图。
- **Stable**: fp16-only，支持 VRAM 优化。

## 🛠️ Installation

### 1. Backend (Python)
- Python 3.10+
- 建议 GPU：3090/4090 (24GB VRAM 推荐，最少 12GB 可运行 FP16)

```bash
# 根目录下
pip install -r requirements.txt
```

### 2. Frontend (Node.js)
- Node.js 18+

```bash
cd frontend
npm install
```

## 📦 Download Models

### SDXL Models
```bash
python scripts/download_models.py --clean
```

### LLM Model (Qwen)
请下载 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 或类似模型至 `models/Qwen...` 目录。

## 🚀 Run Application

你需要开启 **三个终端** 分别运行以下服务：

### 1️⃣ Start LLM Server (Port 8001)
负责古诗词理解与 Prompt 生成。
```bash
# 根目录下
python -m sdxl_app.engine.simple_llm_server --model models/Qwen2.5-1.5B-Instruct --port 8001
```

### 2️⃣ Start Backend Server (Port 8000)
负责 SDXL 图像生成与 Session 管理。
```bash
# 根目录下
python server.py
# 或 python -m sdxl_app.api.server
```

### 3️⃣ Start Frontend (Port 5173)
用户界面。
```bash
cd frontend
npm run dev
```
打开浏览器访问：`http://localhost:5173`

## ⚙️ Configuration

推荐使用环境变量或 YAML 配置。
默认配置文件：`config.py`

```yaml
# 可选：sdxl.yaml
prompts:
  llm_enabled: true
  llm_model: "Qwen2.5-1.5B-Instruct"

models:
  base_path: "models/stable-diffusion-xl-base-1.0"
  inpaint_path: "models/stable-diffusion-xl-1.0-inpainting-0.1"
```

## 🧭 Workflow

1.  **输入诗词**：在输入框输入中文古诗（如“孤舟蓑笠翁”）。
2.  **LLM 解析**：后端自动调用 LLM 解析主体、动作、意境，并生成英文 Prompt。
3.  **生成 (Generate)**：SDXL 生成初版图像。
4.  **编辑 (Edit)**：
    -   在生成的图片上涂抹 Mask。
    -   输入修改指令（如“换成红色衣服”）。
    -   点击 Generate 进行局部重绘。
5.  **历史 (History)**：随时点击下方缩略图回退到任意版本。

## 🧯 Common Issues

-   **LLM Connection Refused**: 请确保 1 号终端 (`simple_llm_server`) 已启动并显示运行在 8001 端口。
-   **CUDA OOM**: 也就是显存不足。
    -   尝试在 `config.py` 中开启 `enable_cpu_offload: true`。
    -   考虑使用更小的 LLM (如 Qwen 0.5B) 或量化版本。
-   **Frontend API Error**: 检查 `frontend/.env` 或代码中的 API 地址是否指向 `http://localhost:8000`。
