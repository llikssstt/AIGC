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
├── frontend/                 # React 前端
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
└── server.py                # 后端入口
```

## 🧯 Common Issues

- **LLM Connection Refused**: 确保 LLM Server 已启动并运行在 8001 端口
- **CUDA OOM**: 显存不足，尝试使用更小的 LLM 或开启 CPU offload
- **Prompt Truncated**: 正常现象，CLIP 限制 77 tokens，系统会自动截断
- **画廊为空**: 需要先创作作品才会显示在画廊中
