# NOTE: legacy entrypoint kept for reference; new UI is `sdxl_app.ui.app`.
# ============================================================
# SDXL Inpainting Editor - Gradio Frontend
# ============================================================
"""
Gradio 前端界面：
- Generate Tab: 文生图
- Edit Tab: 局部编辑
- History: 历史版本管理
"""
import logging
import io
import requests
import os
import tempfile

# -----------------------------------------------------------------------------
# Fix for Gradio 6.0 Issues
# -----------------------------------------------------------------------------
# 1. Bypass proxy issues on Windows
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# 2. Bypass Gradio 6.0 safehttpx hostname validation bug
# This prevents "ValueError: Hostname 127.0.0.1 failed validation"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_ALLOW_FLAGGING"] = "never"

# Gradio 会把上传文件写到临时目录；若 launch(allowed_paths=...) 过窄，会导致上传图片显示空白。
# 这里把 Gradio 临时目录固定到项目内 cache 下，保证 Windows 下上传/粘贴可显示。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
GRADIO_TMP_DIR = os.path.join(CACHE_DIR, "_gradio_tmp")
os.makedirs(GRADIO_TMP_DIR, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", GRADIO_TMP_DIR)

from typing import Optional, List, Dict, Any

import gradio as gr
from PIL import Image

from config import (
    API_HOST, API_PORT, 
    DEFAULT_GENERATE_PARAMS, DEFAULT_EDIT_PARAMS,
    STYLE_PRESETS,
    GRADIO_HOST, GRADIO_PORT
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

API_URL = f"http://{API_HOST}:{API_PORT}"

# ============================================================
# API 客户端函数
# ============================================================
def api_create_session() -> tuple[Optional[str], List[str]]:
    """创建 session"""
    try:
        resp = requests.post(f"{API_URL}/session/create")
        if resp.status_code == 200:
            data = resp.json()
            return data["session_id"], data["available_styles"]
        return None, []
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return None, []

def _load_pil_image(path: str) -> Optional[Image.Image]:
    """安全加载本地图片为 PIL（立即 copy，避免文件句柄占用）。"""
    try:
        if not path:
            return None
        with Image.open(path) as im:
            return im.convert("RGBA").copy()
    except Exception as e:
        logger.error(f"Failed to load image from {path}: {e}")
        return None

def _image_editor_value(background: Optional[Image.Image]) -> Optional[dict]:
    """
    Gradio 6 `gr.ImageEditor` 的 value 需要固定字典结构：
    {"background": <type>, "layers": [<type>], "composite": <type>}
    - 建议更新背景时清空 layers，并把 composite 设为 None（由前端重新合成）
    """
    if background is None:
        return None
    return {"background": background, "layers": [], "composite": None}

def _file_like_to_path(value: Any) -> Optional[str]:
    """
    兼容 Gradio 6 可能返回的 FileData / dict / str。
    - str: 直接返回
    - {"path": "..."}: 返回 path
    - FileData: 取 .path
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"]
    path_attr = getattr(value, "path", None)
    if isinstance(path_attr, str):
        return path_attr
    return None

def _layer_to_pil(layer: Any) -> Optional[Image.Image]:
    """把 ImageEditor layer（PIL / filepath / FileData）统一转成 PIL.Image。"""
    if layer is None:
        return None
    if isinstance(layer, Image.Image):
        return layer
    layer_path = _file_like_to_path(layer)
    if layer_path:
        return _load_pil_image(layer_path)
    return None

def _extract_mask_from_editor(editor_value: Any) -> Optional[Image.Image]:
    """
    从 ImageEditor 的 value 中提取可用于 inpaint 的 L 模式 mask。
    优先使用 RGBA alpha 通道（对“黑色画笔但有透明度”的情况更稳）。
    """
    try:
        if not editor_value or not isinstance(editor_value, dict):
            return None
        layers = editor_value.get("layers") or []
        if not isinstance(layers, list) or len(layers) == 0:
            return None

        import numpy as np

        combined: Optional[np.ndarray] = None
        for layer in layers:
            layer_img = _layer_to_pil(layer)
            if layer_img is None:
                continue

            rgba = layer_img.convert("RGBA")
            alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
            # 有些情况下 layer 会是“整张不透明图”，alpha 全 255，此时用 alpha 会把整图当作 mask
            if alpha.min() == alpha.max():
                gray = np.array(rgba.convert("L"), dtype=np.uint8)
                candidate = gray
            else:
                candidate = alpha
            if combined is None:
                combined = candidate
            else:
                combined = np.maximum(combined, candidate)

        if combined is None:
            return None
        return Image.fromarray(combined, mode="L")
    except Exception as e:
        logger.error(f"Failed to extract mask from ImageEditor value: {e}")
        return None

def _download_to_file(url: str, suffix: str = ".png") -> Optional[str]:
    """从 URL 下载图片保存到本地文件，返回绝对路径"""
    try:
        # 使用 hash 命名避免冲突
        import hashlib
        filename = hashlib.md5(url.encode()).hexdigest() + suffix
        filepath = os.path.join(CACHE_DIR, filename)
        
        # 如果文件已存在且大小不为0，直接返回
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return os.path.abspath(filepath)
            
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return os.path.abspath(filepath)
        return None
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return None

def api_generate(
    session_id: str,
    style: str,
    scene_text: str,
    seed: int,
    steps: int,
    cfg: float,
    height: int,
    width: int
) -> tuple[Optional[str], Optional[dict]]:
    """调用生成 API，返回本地文件路径"""
    try:
        payload = {
            "style_preset": style,
            "scene_text": scene_text,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "height": height,
            "width": width
        }
        resp = requests.post(f"{API_URL}/session/{session_id}/generate", json=payload, timeout=600)
        if resp.status_code == 200:
            data = resp.json()
            image_url = f"{API_URL}{data['image_url']}"
            local_path = _download_to_file(image_url)
            return local_path, data["prompt_card"]
        logger.error(f"Generate error: {resp.text}")
        return None, None
    except Exception as e:
        logger.error(f"Generate exception: {e}")
        return None, None

def api_edit(
    session_id: str,
    image_dict: dict,
    edit_text: str,
    params: dict
) -> tuple[Optional[str], Optional[dict]]:
    """调用编辑 API，返回本地文件路径"""
    try:
        # image_dict from Gradio ImageEditor:
        # {"background": <PIL|filepath|FileData>, "layers": [...], "composite": ...}
        mask_img = _extract_mask_from_editor(image_dict)
        if mask_img is None:
            logger.error("No mask drawn (layers empty or mask extraction failed)")
            return None, None

        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        mask_bytes = buf.getvalue()
        
        files = {
            "mask": ("mask.png", mask_bytes, "image/png")
        }
        
        data = {
            "edit_text": edit_text,
            **params
        }
        
        resp = requests.post(
            f"{API_URL}/session/{session_id}/edit",
            data=data,
            files=files,
            timeout=600,
        )
        
        if resp.status_code == 200:
            data = resp.json()
            image_url = f"{API_URL}{data['image_url']}"
            local_path = _download_to_file(image_url)
            return local_path, data["prompt_card"]
            
        logger.error(f"Edit error: {resp.text}")
        return None, None
        
    except Exception as e:
        logger.error(f"Edit exception: {e}")
        return None, None

def api_get_history(session_id: str) -> List[tuple[str, str]]:
    """获取历史记录，返回本地路径"""
    try:
        resp = requests.get(f"{API_URL}/session/{session_id}/history", timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            result = []
            for item in data:
                thumb_url = f"{API_URL}{item['thumbnail_url']}"
                local_path = _download_to_file(thumb_url, suffix="_thumb.png")
                if local_path:
                    result.append((local_path, f"v{item['version']} - {item['edit_type']}"))
            return result
        return []
    except Exception:
        return []

def api_revert(session_id: str, version: int):
    """回退版本"""
    try:
        requests.post(f"{API_URL}/session/{session_id}/revert", data={"version": version}, timeout=60)
    except Exception:
        pass

def api_get_image(session_id: str, version: int) -> Optional[str]:
    """获取图片，返回本地路径"""
    url = f"{API_URL}/session/{session_id}/image/{version}"
    return _download_to_file(url)


# ============================================================
# Gradio 界面逻辑
# ============================================================
with gr.Blocks(title="SDXL Inpainting Editor") as demo:
    # 状态存储
    session_id_state = gr.State(value=None)
    current_version_state = gr.State(value=-1)
    
    gr.Markdown("# 🎨 SDXL Inpainting & Text-Driven Editor")
    
    with gr.Row():
        # 左侧：主要操作区
        with gr.Column(scale=3):
            with gr.Tabs() as tabs:
                # Tab 1: Generate (文生图)
                with gr.TabItem("Generate v0", id="tab_generate"):
                    with gr.Row():
                        style_dropdown = gr.Dropdown(
                            choices=list(STYLE_PRESETS.keys()),
                            value="水墨",
                            label="Style Preset"
                        )
                        scene_input = gr.Textbox(
                            label="Scene Description",
                            placeholder="Describe the scene (e.g., misty mountains, lone pine tree)..."
                        )
                    
                    with gr.Accordion("Advanced Parameters", open=False):
                        with gr.Row():
                            seed_gen = gr.Number(label="Seed", value=-1, precision=0)
                            steps_gen = gr.Slider(label="Steps", minimum=10, maximum=100, value=30, step=1)
                            cfg_gen = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.1)
                        with gr.Row():
                            width_gen = gr.Slider(label="Width", minimum=512, maximum=1536, value=1024, step=64)
                            height_gen = gr.Slider(label="Height", minimum=512, maximum=1536, value=1024, step=64)
                    
                    btn_generate = gr.Button("Generate Initial Image", variant="primary")
                
                # Tab 2: Edit (局部编辑)
                with gr.TabItem("Edit", id="tab_edit"):
                    gr.Markdown("### Draw mask on the image to edit")
                    # ImageEditor: supports drawing mask
                    image_editor = gr.ImageEditor(
                        label="Editor",
                        type="pil",
                        height=600,
                        sources=["upload", "clipboard"],
                    )
                    
                    with gr.Row():
                        edit_text_input = gr.Textbox(
                            label="Edit Instruction",
                            placeholder="What to change? (e.g., add a red sun, remove the rock)...",
                            scale=4
                        )
                        btn_edit = gr.Button("Apply Edit", variant="primary", scale=1)
                    
                    with gr.Accordion("Edit Parameters", open=False):
                        with gr.Row():
                            strength_slider = gr.Slider(label="Strength (Denoise)", minimum=0.0, maximum=1.0, value=0.6, step=0.01)
                            grow_slider = gr.Slider(label="Mask Grow (px)", minimum=0, maximum=50, value=8, step=1)
                            blur_slider = gr.Slider(label="Mask Blur (sigma)", minimum=0.0, maximum=20.0, value=12.0, step=0.5)
                        with gr.Row():
                            seed_edit = gr.Number(label="Seed (-1 random)", value=-1, precision=0)
                            steps_edit = gr.Slider(label="Steps", minimum=10, maximum=100, value=30, step=1)
                            cfg_edit = gr.Slider(label="CFG", minimum=1.0, maximum=20.0, value=7.5, step=0.1)
                            invert_check = gr.Checkbox(label="Invert Mask", value=False)
        
        # 右侧：历史与信息
        with gr.Column(scale=1):
            gr.Markdown("### History")
            # Gallery to show history thumbnails
            history_gallery = gr.Gallery(
                label="Versions",
                columns=2,
                rows=2,
                object_fit="contain",
                height=300,
                allow_preview=False
            )
            
            with gr.Row():
                btn_revert = gr.Button("Revert / Continue", variant="secondary")
                btn_params = gr.JSON(label="Current Params", visible=False)
            
            gr.Markdown("### Prompt Card")
            prompt_card_display = gr.JSON(label="Detailed Info")
            
            # Helper text box for copying prompt
            final_prompt_display = gr.Textbox(label="Final Prompt", interactive=False, lines=3)

    # ============================================================
    # 事件处理
    # ============================================================
    
    # 1. App 加载时创建 session
    def init_session():
        sid, _ = api_create_session()
        return sid, []

    demo.load(init_session, outputs=[session_id_state, history_gallery])

    # 2. Generate 按钮点击
    def on_generate(sid, style, text, seed, steps, cfg, width, height):
        if not sid:
            sid, _ = api_create_session()
        
        image_path, card = api_generate(sid, style, text, seed, steps, cfg, height, width)
        
        if image_path:
            bg = _load_pil_image(image_path)
            editor_value = _image_editor_value(bg)
            return (
                sid,
                editor_value,  # 设置 ImageEditor 背景
                card,
                card["final_prompt"] if card else "",
                gr.Tabs(selected="tab_edit"),
                # 刷新历史
                api_get_history(sid)
            )
        return sid, None, None, "Error generating image", gr.Tabs(selected="tab_generate"), []

    btn_generate.click(
        on_generate,
        inputs=[session_id_state, style_dropdown, scene_input, seed_gen, steps_gen, cfg_gen, width_gen, height_gen],
        outputs=[session_id_state, image_editor, prompt_card_display, final_prompt_display, tabs, history_gallery]
    )

    # 3. Edit 按钮点击
    def on_edit(sid, img_dict, text, strength, grow, blur, seed, steps, cfg, invert):
        params = {
            "strength": strength,
            "grow_pixels": grow,
            "blur_sigma": blur,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "invert_mask": invert
        }
        
        image_path, card = api_edit(sid, img_dict, text, params)
        
        if image_path:
            bg = _load_pil_image(image_path)
            editor_value = _image_editor_value(bg)
            return (
                editor_value,
                card,
                card["final_prompt"] if card else "",
                api_get_history(sid)
            )
        return None, None, "Error editing image", []

    btn_edit.click(
        on_edit,
        inputs=[
            session_id_state, image_editor, edit_text_input,
            strength_slider, grow_slider, blur_slider,
            seed_edit, steps_edit, cfg_edit, invert_check
        ],
        outputs=[image_editor, prompt_card_display, final_prompt_display, history_gallery]
    )

    # 4. 历史记录点击 (Revert)
    def on_select_history(evt: gr.SelectData, sid, history_list):
        # evt.index 是 gallery 中的索引
        # 我们需要找到对应的 version。
        version = evt.index
        api_revert(sid, version)
        
        image_path = api_get_image(sid, version)
        if image_path:
            bg = _load_pil_image(image_path)
            return _image_editor_value(bg)
        return None

    history_gallery.select(
        on_select_history,
        inputs=[session_id_state, history_gallery],
        outputs=[image_editor]
    )

    # 5. 回退按钮 (Revert/Continue from selected)
    # 其实 on_select_history 已经做了这件事：把选中的图放到 ImageEditor
    # 用户可以在这基础上继续编辑。
    # 这里的按钮可以是 "刷新历史" 或 "新建会话"
    
    btn_revert.click(
        lambda sid: api_get_history(sid),
        inputs=[session_id_state],
        outputs=[history_gallery] 
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=GRADIO_HOST, 
        server_port=GRADIO_PORT, 
        share=False,
        allowed_paths=[CACHE_DIR, GRADIO_TMP_DIR, tempfile.gettempdir()],
        theme=gr.themes.Soft()
    )
