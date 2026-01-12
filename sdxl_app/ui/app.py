from __future__ import annotations

import io
import logging
import os
from typing import Any, Optional

import gradio as gr
import numpy as np
import requests
from PIL import Image

from sdxl_app.config import get_settings

logger = logging.getLogger(__name__)


def _requests_session() -> requests.Session:
    # Windows 上 requests 会读取系统代理，导致本地 loopback 访问异常；这里强制忽略环境代理。
    session = requests.Session()
    session.trust_env = False
    return session


def _image_editor_value(background: Optional[Image.Image]) -> Optional[dict]:
    if background is None:
        return None
    bg = background.convert("RGB")
    return {"background": bg, "layers": [], "composite": bg}


def _file_like_to_path(value: Any) -> Optional[str]:
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
    if layer is None:
        return None
    if isinstance(layer, Image.Image):
        return layer
    layer_path = _file_like_to_path(layer)
    if not layer_path:
        return None
    try:
        with Image.open(layer_path) as im:
            return im.convert("RGBA").copy()
    except Exception:
        return None


def _extract_mask_from_editor(editor_value: Any) -> Optional[Image.Image]:
    """
    Gradio ImageEditor value: {"background": <...>, "layers": [...], "composite": ...}
    合成所有 layer 的 alpha（或灰度）为单通道 mask。
    """
    if not editor_value or not isinstance(editor_value, dict):
        return None
    layers = editor_value.get("layers") or []
    if not isinstance(layers, list) or len(layers) == 0:
        return None

    combined: Optional[np.ndarray] = None
    for layer in layers:
        layer_img = _layer_to_pil(layer)
        if layer_img is None:
            continue

        rgba = layer_img.convert("RGBA")
        alpha = np.array(rgba.getchannel("A"), dtype=np.uint8)
        if alpha.min() == alpha.max():
            gray = np.array(rgba.convert("L"), dtype=np.uint8)
            candidate = gray
        else:
            candidate = alpha

        combined = candidate if combined is None else np.maximum(combined, candidate)

    if combined is None:
        return None
    return Image.fromarray(combined, mode="L")


def _fetch_image(session: requests.Session, url: str, timeout: int = 120) -> Image.Image:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def build_demo() -> gr.Blocks:
    settings = get_settings()
    api_url = settings.api_url
    http = _requests_session()

    def api_create_session() -> str:
        r = http.post(f"{api_url}/session/create", timeout=30)
        r.raise_for_status()
        return r.json()["session_id"]

    def api_generate(
        session_id: str,
        style: str,
        scene_text: str,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
    ) -> dict:
        payload = {
            "style_preset": style,
            "scene_text": scene_text,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "width": int(width),
            "height": int(height),
        }
        r = http.post(f"{api_url}/session/{session_id}/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    def api_import_image(session_id: str, image: Image.Image) -> dict:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        files = {"image": ("image.png", buf.getvalue(), "image/png")}
        r = http.post(f"{api_url}/session/{session_id}/import", files=files, timeout=120)
        r.raise_for_status()
        return r.json()

    def api_edit(session_id: str, mask_png: bytes, edit_text: str, params: dict) -> dict:
        files = {"mask": ("mask.png", mask_png, "image/png")}
        data = {"edit_text": edit_text, **params}
        r = http.post(f"{api_url}/session/{session_id}/edit", data=data, files=files, timeout=600)
        r.raise_for_status()
        return r.json()

    def api_history(session_id: str) -> list[dict]:
        r = http.get(f"{api_url}/session/{session_id}/history", timeout=60)
        r.raise_for_status()
        return r.json()

    def api_revert(session_id: str, version: int) -> None:
        r = http.post(f"{api_url}/session/{session_id}/revert", data={"version": int(version)}, timeout=30)
        r.raise_for_status()

    with gr.Blocks(title="SDXL Local Editor") as demo:
        session_id = gr.State(value=None)
        history_versions = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=3):
                tabs = gr.Tabs()
                with gr.TabItem("Generate", id="tab_generate"):
                    style = gr.Dropdown(
                        choices=list(settings.prompts.style_presets.keys()),
                        value=list(settings.prompts.style_presets.keys())[0],
                        label="Style",
                    )
                    scene = gr.Textbox(
                        label="Scene / Poem",
                        placeholder="输入一句话或一首诗（支持多行）。系统会自动抽取意象并增强提示词。",
                        lines=4,
                    )
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                    steps = gr.Slider(10, 80, value=settings.defaults.generate_steps, step=1, label="Steps")
                    cfg = gr.Slider(1.0, 20.0, value=settings.defaults.generate_cfg, step=0.1, label="CFG")
                    width = gr.Slider(512, 1536, value=settings.defaults.generate_width, step=64, label="Width")
                    height = gr.Slider(512, 1536, value=settings.defaults.generate_height, step=64, label="Height")
                    btn_gen = gr.Button("Generate", variant="primary")

                with gr.TabItem("Edit", id="tab_edit"):
                    import_image = gr.Image(label="Import Base Image (optional)", type="pil")
                    btn_import = gr.Button("Set As Current Base", variant="secondary")

                    editor = gr.ImageEditor(
                        label="Editor",
                        type="pil",
                        height=600,
                        sources=["clipboard"],  # 避免 Windows 下 ImageEditor upload 空白的问题；用上面的 gr.Image 导入
                    )
                    edit_text = gr.Textbox(label="Edit Instruction")
                    strength = gr.Slider(0.0, 1.0, value=settings.defaults.edit_strength, step=0.01, label="Strength")
                    grow = gr.Slider(0, 50, value=settings.defaults.edit_grow_pixels, step=1, label="Mask Grow")
                    blur = gr.Slider(0.0, 20.0, value=settings.defaults.edit_blur_sigma, step=0.5, label="Mask Blur")
                    invert = gr.Checkbox(value=settings.defaults.edit_invert_mask, label="Invert Mask")
                    seed_e = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                    steps_e = gr.Slider(10, 80, value=settings.defaults.edit_steps, step=1, label="Steps")
                    cfg_e = gr.Slider(1.0, 20.0, value=settings.defaults.edit_cfg, step=0.1, label="CFG")
                    btn_edit = gr.Button("Apply Edit", variant="primary")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="History", columns=2, height=320, allow_preview=False)
                card = gr.JSON(label="Prompt Card")

        def ensure_session(sid: Optional[str]) -> str:
            return sid or api_create_session()

        def refresh_history(sid: str):
            items = api_history(sid)
            versions = [it["version"] for it in items]
            thumbs = []
            for it in items:
                thumb = _fetch_image(http, f"{api_url}{it['thumbnail_url']}", timeout=120)
                thumbs.append((thumb, f"v{it['version']} {it['edit_type']}"))
            return thumbs, versions

        def on_generate(sid, style, scene, seed, steps, cfg, width, height):
            sid = ensure_session(sid)
            resp = api_generate(sid, style, scene, seed, steps, cfg, width, height)
            img = _fetch_image(http, f"{api_url}{resp['image_url']}", timeout=600)
            thumbs, versions = refresh_history(sid)
            return sid, _image_editor_value(img), resp["prompt_card"], thumbs, versions, gr.Tabs(selected="tab_edit")

        def on_import(sid, image: Image.Image):
            sid = ensure_session(sid)
            if image is None:
                raise gr.Error("Please upload an image first.")
            resp = api_import_image(sid, image)
            img = _fetch_image(http, f"{api_url}{resp['image_url']}", timeout=120)
            thumbs, versions = refresh_history(sid)
            return sid, _image_editor_value(img), resp["prompt_card"], thumbs, versions

        def on_edit(sid, editor_value, edit_text, strength, grow, blur, invert, seed, steps, cfg):
            if not sid:
                raise gr.Error("Session not created")
            mask_img = _extract_mask_from_editor(editor_value)
            if mask_img is None:
                raise gr.Error("Please draw a mask before editing")

            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            params = {
                "strength": float(strength),
                "grow_pixels": int(grow),
                "blur_sigma": float(blur),
                "invert_mask": bool(invert),
                "seed": int(seed),
                "steps": int(steps),
                "cfg": float(cfg),
            }
            resp = api_edit(sid, buf.getvalue(), edit_text, params)
            img = _fetch_image(http, f"{api_url}{resp['image_url']}", timeout=600)
            thumbs, versions = refresh_history(sid)
            return _image_editor_value(img), resp["prompt_card"], thumbs, versions

        def on_select(evt: gr.SelectData, sid: str, versions: list[int]):
            if not sid:
                return None, None
            if not versions or evt.index is None:
                return None, None
            v = versions[int(evt.index)]
            api_revert(sid, v)
            img = _fetch_image(http, f"{api_url}/session/{sid}/image/{v}", timeout=120)
            return _image_editor_value(img), None

        btn_gen.click(
            on_generate,
            inputs=[session_id, style, scene, seed, steps, cfg, width, height],
            outputs=[session_id, editor, card, gallery, history_versions, tabs],
        )
        btn_import.click(
            on_import,
            inputs=[session_id, import_image],
            outputs=[session_id, editor, card, gallery, history_versions],
        )
        btn_edit.click(
            on_edit,
            inputs=[session_id, editor, edit_text, strength, grow, blur, invert, seed_e, steps_e, cfg_e],
            outputs=[editor, card, gallery, history_versions],
        )
        gallery.select(on_select, inputs=[session_id, history_versions], outputs=[editor, card])

    return demo


def main() -> None:
    settings = get_settings()
    demo = build_demo()

    # 避免 Windows 上 requests 走系统代理导致 loopback 失败
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,0.0.0.0")

    demo.queue().launch(
        server_name=settings.ui.host,
        server_port=settings.ui.port,
        share=settings.ui.share,
    )


if __name__ == "__main__":
    main()
