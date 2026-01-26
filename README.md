# 古诗·绘意 Digital Ink & Poetry

本项目是一个本地 SDXL 应用，结合了 **React 前端**、**FastAPI 后端** 和 **本地 LLM (Qwen)**，支持：
文生图 (古诗理解) → 标注 mask → 多轮局部编辑 → 历史回退 → 画廊浏览。

## ✨ Features

- **🎨 三种国风风格**: 水墨、工笔、青绿，一键切换
- **📝 Prompt Optimization**: 内置 LLM (Qwen) 理解古诗词，自动生成结构化 prompt
- **🖼️ Generate + Inpaint**: 同一套 API 支持 `text2img` 与 `inpaint`
- **✏️ Canvas Mask Editor**: 可视化蒙版绘制，支持膨胀/羽化/反转
- **📚 Gallery**: 画廊展示所有历史作品，支持继续编辑
- **🔄 Session & History**: 多轮版本、回退、缩略图
- **⚙️ Advanced Parameters**: 可调 Seed / Steps / CFG / Strength 等参数
- **🚀 一键启动**: PowerShell 脚本同时启动所有服务

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
请下载 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 或类似模型至 `models/` 目录。

### Lora Models
请下载 [Lora模型](https://huggingface.co/Hiwebsun0914/stable-diffusion-xl-base-1.0-unet-lora)至 `models/stable-diffusion-xl-base-1.0/`目录下。

## 🚀 Run Application

### 一键启动 (推荐)
```powershell
.\start_all.ps1
```
该脚本会自动打开三个终端窗口运行所有服务。

### 手动启动
如需单独启动各服务：

**1️⃣ LLM Server (Port 8001)**
```bash
python -m sdxl_app.engine.simple_llm_server --model models/Qwen3-1.7B --port 8001
```

**2️⃣ Backend Server (Port 8000)**
```bash
python server.py
```

**3️⃣ Frontend (Port 5173)**
```bash
cd frontend
npm run dev
```

打开浏览器访问：`http://localhost:5173`

## 🧭 Workflow

1. **选择风格**: 水墨 / 工笔 / 青绿
2. **输入诗词**: 在输入框输入中文古诗（如"孤舟蓑笠翁，独钓寒江雪"）
3. **调整参数** (可选): 展开"高级参数"调整 Seed、Steps、CFG 等
4. **生成**: 点击"生成意境"，等待 SDXL 生成图像
5. **编辑**: 点击"编辑此图"进入编辑模式
   - 涂抹需要修改的区域（红色蒙版）
   - 输入修改指令（如"换成红色衣服"）
   - 调整 Strength（0.3-0.5 微调，0.7-0.9 大改）
   - 点击"应用修改"
6. **查看历史**: 点击"历史"查看/回退到任意版本
7. **画廊**: 在首页点击"画廊"浏览所有历史作品

## 📁 Project Structure

```
AIGC/
├── frontend/                # React 前端
│   └── src/
│       ├── pages/           # 页面组件 (Creation, Edit, Gallery)
│       ├── components/      # UI 组件 (InkButton, MaskCanvas, etc.)
│       └── services/        # API 服务
├── sdxl_app/                # 后端核心
│   ├── api/server.py        # FastAPI 路由
│   ├── engine/              # SDXL 引擎 + LLM 服务 + Prompt 编译
│   └── storage/             # Session 存储管理
├── models/                  # 模型文件 (SDXL, Qwen)
├── storage/sessions/        # 生成的图片和元数据
├── start_all.ps1            # 一键启动脚本
├── scripts                  # LoRA训练代码
└── server.py                # 后端入口
```

## 🧯 Common Issues

- **LLM Connection Refused**: 确保 LLM Server 已启动并运行在 8001 端口
- **CUDA OOM**: 显存不足，尝试使用更小的 LLM 或开启 CPU offload
- **Prompt Truncated**: 正常现象，CLIP 限制 77 tokens，系统会自动截断
- **画廊为空**: 需要先创作作品才会显示在画廊中

## 附：LoRA训练说明

项目默认提供一份 LoRA（diffusers 格式）在：
- `models/stable-diffusion-xl-base-1.0/unet_lora/`

后端读取环境变量（前缀为 `SDXL_`）：
- `SDXL_MODELS_LORA_PATH`：LoRA 路径（**目录**或单个 `.safetensors`）
- `SDXL_MODELS_LORA_SCALE`：强度（常用 `0.5 ~ 1.0`）
- `SDXL_MODELS_LORA_FUSE`：是否 fuse（`True/False`）

示例（PowerShell）：

```powershell
$env:SDXL_MODELS_LORA_PATH = "models/stable-diffusion-xl-base-1.0/unet_lora"
$env:SDXL_MODELS_LORA_SCALE = "0.8"
$env:SDXL_MODELS_LORA_FUSE = "True"
python -m sdxl_app.api.server
```

关闭 LoRA：不要设置 `SDXL_MODELS_LORA_PATH`（或在 `start_all.ps1` 里移除该环境变量）。


核心训练脚本：`scripts/lora_finetune.py`

它做三件事：
1) 扫描分类后的数据集；
2) 生成/同步图片 captions（存到 `LoRA/captions.csv`）；
3) 训练并导出 SDXL UNet LoRA（diffusers 格式 `unet_lora/`）。


训练脚本要求数据集结构如下（风格目录名必须一致）：

```
Chinese-Landscape-Painting-Dataset/
  sorted_by_style/
    水墨/
    工笔/
    青绿/
```

请前往 [国风数据集](https://huggingface.co/datasets/Hiwebsun0914/Chinese-Painting) 下载`Chinese-Landscape-Painting-Dataset/sorted_by_style`到根目录


在准备数据标签时，先启动 Qwen 服务（默认端口 8001）：

```bash
python -m sdxl_app.engine.simple_llm_server --model "<QWEN_PATH>" --port 8001
```

生成/刷新标签（只更新 CSV，不训练）：

```powershell
python scripts/lora_finetune.py `
  --dataset ".\\Chinese-Landscape-Painting-Dataset\\sorted_by_style" `
  --caption-table "LoRA\\captions.csv" `
  --caption-only
```

`LoRA/captions.csv` 字段为：
- `style`：水墨/工笔/青绿
- `relative_path`：相对 `sorted_by_style` 的路径
- `caption`：训练用描述文本
- `last_updated`：时间戳

训练（LoRA fine-tune）

```powershell
python scripts/lora_finetune.py `
  --dataset ".\\Chinese-Landscape-Painting-Dataset\\sorted_by_style" `
  --caption-table "LoRA\\captions.csv" `
  --lora-dir "LoRA" `
  --pretrained-model "models\\stable-diffusion-xl-base-1.0" `
  --checkpoint-name "style_adapter" `
  --batch-size 4 `
  --epochs 1 `
  --resolution 512 `
  --learning-rate 2e-4 `
  --save-steps 100 `
  --fp16 `
  --num-workers 0
```

输出位置：
- `LoRA/style_adapter_stepXXXX/unet_lora/`