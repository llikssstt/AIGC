# ============================================================
# SDXL Inpainting Editor - Configuration
# ============================================================
"""
配置文件：模型路径、设备、默认参数、风格预设
"""
from typing import Dict, Any
from pathlib import Path

# ============================================================
# 模型路径配置
# ============================================================

# --- 选项 A: SDXL (高画质, ~12GB) ---
MODEL_PATH_TEXT2IMG: str = r"models/stable-diffusion-xl-base-1.0"
MODEL_PATH_INPAINT: str = r"models/stable-diffusion-xl-1.0-inpainting-0.1"

# --- 选项 B: DreamShaper 8 (轻量级/快, ~4GB) ---
# 只需下载轻量模型并取消下方注释即可切换
# MODEL_PATH_TEXT2IMG: str = r"models/dreamshaper-8"
# MODEL_PATH_INPAINT: str = r"models/dreamshaper-8-inpainting"

# 可选：Refiner 模型路径（设为 None 禁用）
MODEL_PATH_REFINER: str | None = None

# ============================================================
# 设备与精度配置
# ============================================================
DEVICE: str = "cuda"  # "cuda" 或 "cpu"
TORCH_DTYPE: str = "float16"  # "float16" 或 "float32"，float16 节省显存

# ============================================================
# 存储路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage" / "sessions"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 风格预设 - 国风三套
# ============================================================
STYLE_PRESETS: Dict[str, str] = {
    "水墨": (
        "traditional Chinese ink wash painting, shuimo style, "
        "flowing ink strokes, elegant brushwork, monochrome with subtle color gradients, "
        "xieyi freehand style, rice paper texture, "
        "highly detailed, masterpiece, best quality"
    ),
    "工笔": (
        "Chinese gongbi meticulous painting style, fine brushwork, "
        "delicate lines, rich colors, intricate details, "
        "silk painting texture, traditional pigments, "
        "highly detailed, masterpiece, best quality"
    ),
    "青绿": (
        "Chinese qinglv style painting, blue and green landscape painting, "
        "mineral pigments, azurite blue and malachite green, "
        "Tang dynasty aesthetic, golden outlines, "
        "decorative style, highly detailed, masterpiece, best quality"
    ),
}

# ============================================================
# 默认负面提示词
# ============================================================
DEFAULT_NEGATIVE_PROMPT: str = (
    "watermark, text, logo, signature, username, "
    "low quality, worst quality, blurry, pixelated, "
    "deformed, ugly, bad anatomy, extra limbs, "
    "photorealistic, 3d render, photography, "
    "modern elements, western style, "
    "nsfw, nude,"
    "empty landscape, scenery only, no humans, no characters, " # 强制禁止空景
    "calligraphy, chinese characters, poem text, " # 防止AI试图把诗句写在画上（通常写得很丑）
    "overexposed, underexposed"
)

# Inpainting 专用负面提示词（额外约束）
INPAINT_NEGATIVE_APPEND: str = (
    ", change background, outside mask changes, "
    "extra objects outside region, affect unmasked area, "
    "inconsistent style, different lighting"
)

# ============================================================
# 默认生成参数
# ============================================================
DEFAULT_GENERATE_PARAMS: Dict[str, Any] = {
    "steps": 30,
    "cfg": 7.5,
    "height": 1024,
    "width": 1024,
    "seed": -1,  # -1 表示随机
}

# ============================================================
# 默认编辑参数
# ============================================================
DEFAULT_EDIT_PARAMS: Dict[str, Any] = {
    "steps": 30,
    "cfg": 7.5,
    "strength": 0.85,  # 替换主体推荐 0.75-0.90，微调推荐 0.35-0.55
    "seed": -1,
    "grow_pixels": 8,      # mask 膨胀像素
    "blur_sigma": 12.0,    # 羽化高斯模糊 sigma
    "invert_mask": False,  # 是否反转 mask
}

# ============================================================
# API 服务配置
# ============================================================
API_HOST: str = "127.0.0.1"  # 本地访问，如需外网访问可改为 "0.0.0.0"
API_PORT: int = 8000

# Gradio 配置
GRADIO_HOST: str = "127.0.0.1"
GRADIO_PORT: int = 7860
