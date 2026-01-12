# SDXL Inpainting & Multi-Round Editor (Local)

本项目是一个本地 SDXL 应用，支持：文生图 → 标注 mask → 多轮局部编辑 → 历史回退，并为每次生成/编辑输出可复现的参数卡片（JSON）。

## ✨ Features

- **Generate + Inpaint**: 同一套 API 支持 `text2img` 与 `inpaint`
- **Mask Processing**: grow / feather / invert + alpha blend 后融合
- **Prompt Card**: 统一 prompt 编译与可复现卡片（JSON）
- **Session & History**: 多轮版本、回退、缩略图
- **Stable Download**: fp16-only 下载策略 + 清理 + sanity check
- **UI**: Gradio（默认避免 URL/本地路径喂组件，规避 Gradio 6 SSRF/路径问题）

## 🛠️ Installation

- Python 3.10+
- 建议 GPU：3090/A100；默认 fp16；支持 CPU fallback（慢）

```bash
pip install -r requirements.txt
```

可选：运行单元测试
```bash
pip install -r requirements-dev.txt
```

## 📦 Download Models (fp16-only)

```bash
python scripts/download_models.py --clean
```

脚本会自动进行 sanity check，并给出环境变量/YAML 配置示例。

## ⚙️ Configuration (env / yaml)

推荐用环境变量覆盖（Windows PowerShell 示例）：
```powershell
$env:SDXL_MODELS_BASE_PATH="E:\AIGC\sdxl\models\stable-diffusion-xl-base-1.0"
$env:SDXL_MODELS_INPAINT_PATH="E:\AIGC\sdxl\models\stable-diffusion-xl-1.0-inpainting-0.1"
```

或使用 YAML：
```yaml
models:
  base_path: "models/stable-diffusion-xl-base-1.0"
  inpaint_path: "models/stable-diffusion-xl-1.0-inpainting-0.1"
runtime:
  device: "cuda"
  dtype: "fp16"
```

然后：
```bash
# Windows CMD: set SDXL_CONFIG=sdxl.yaml
# PowerShell: $env:SDXL_CONFIG="sdxl.yaml"
```

## 🚀 Run

### 1) Start Backend (FastAPI)
```bash
python server.py
```
默认：`http://127.0.0.1:8000`

### 2) Start UI (Gradio)
```bash
python app.py
```
默认：`http://127.0.0.1:7860`

## 🧭 Workflow

1. Generate：选择风格 + 场景描述 → 生成 v0
2. Edit：在 Editor 上画 mask → 输入 edit instruction → 多轮编辑生成 v1/v2...
3. History：点击缩略图回退 → 在旧版本继续编辑
4. Import（可选）：用 `Import Base Image` 上传任意图片作为当前 base（解决 ImageEditor upload 在 Windows 下不稳定的问题）

## ✅ Tests

```bash
pytest -q
```

## 📁 Project Structure (new)

```
sdxl/
├─ app.py                      # Entry (Gradio) -> sdxl_app.ui.app
├─ server.py                   # Entry (FastAPI) -> sdxl_app.api.server
├─ sdxl_app/
│  ├─ config.py                # env/yaml 统一配置 + 日志
│  ├─ api/server.py            # FastAPI 路由
│  ├─ engine/                  # 推理引擎 + prompt/mask
│  ├─ storage/session_store.py # session/version 存储层
│  └─ ui/app.py                # Gradio UI
├─ scripts/download_models.py  # fp16-only 下载脚本
├─ tests/                      # 单测骨架
└─ legacy/                     # 旧版 app/server（保留参考）
```

## 🧯 Common Issues

- **Gradio 6 SSRF / 127.0.0.1 validation**：UI 端不把 URL 直接喂给组件，统一由 Python `requests` 拉取后以 PIL 更新组件。
- **Windows 代理导致本地请求失败**：建议关闭系统代理，或设置 `NO_PROXY=localhost,127.0.0.1`。
- **CUDA OOM**：降低分辨率/steps；开启 cpu offload；必要时改 `SDXL_RUNTIME_DTYPE=fp32`（更慢更耗显存）。
- **xformers 缺失**：不影响功能，只是性能下降；可按你的 CUDA/torch 版本安装匹配的 xformers。
