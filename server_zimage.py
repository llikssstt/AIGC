"""
Z-Image-Turbo + SDXL Inpaint 混合服务器
- 文生图: Z-Image-Turbo (8步极速)
- 编辑/Inpaint: SDXL Inpainting

使用方法:
    python server_zimage.py --zimage-model models/Z-Image-Turbo-BF16

或者直接运行 (使用默认路径):
    python server_zimage.py
"""
from __future__ import annotations

import argparse
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from sdxl_app.engine.zimage_engine import ZImageHybridEngine, GenerateRequest, InpaintRequest
from sdxl_app.engine.prompt_utils import PromptCompiler
from sdxl_app.engine.mask_utils import MaskProcessor
from sdxl_app.engine.llm_service import get_llm_service
from sdxl_app.storage.session_store import SessionManager
from sdxl_app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# === Request/Response Models ===

class CreateSessionResponse(BaseModel):
    session_id: str
    available_styles: list[str]


class GenerateRequestModel(BaseModel):
    style_preset: str
    scene_text: str
    seed: int = -1
    steps: int = 8  # Z-Image 默认 8 步
    cfg: float = 3.5  # Z-Image 推荐 3.5
    height: int = 1024
    width: int = 1024


class GenerateResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict


class EditResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict


def create_app(zimage_model_path: str, dtype: str = "bf16") -> FastAPI:
    settings = get_settings()

    session_mgr = SessionManager(settings.storage.sessions_dir)
    
    # LLM Service
    llm_service = None
    if settings.prompts.llm_enabled:
        llm_service = get_llm_service(
            base_url=settings.prompts.llm_base_url,
            model=settings.prompts.llm_model,
            api_key=settings.prompts.llm_api_key,
            timeout=settings.prompts.llm_timeout,
        )

    prompt_compiler = PromptCompiler(
        style_presets=settings.prompts.style_presets,
        negative_prompt=settings.prompts.negative_prompt,
        inpaint_negative_append=settings.prompts.inpaint_negative_append,
        poetry_enabled=settings.prompts.poetry_enabled,
        poetry_negative_append=settings.prompts.poetry_negative_append,
        llm_service=llm_service,
    )
    mask_processor = MaskProcessor()
    
    # 混合引擎: Z-Image + SDXL Inpaint
    engine = ZImageHybridEngine(
        zimage_path=zimage_model_path,
        inpaint_path=settings.models.inpaint_path,
        device=settings.runtime.device,
        dtype=dtype,  # 支持 fp8, bf16, fp16, fp32
        enable_cpu_offload=settings.runtime.enable_cpu_offload,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Z-Image Hybrid Server starting...")
        logger.info("  - Text2Img: Z-Image-Turbo (%s)", zimage_model_path)
        logger.info("  - Inpaint: SDXL (%s)", settings.models.inpaint_path)
        yield
        engine.unload()
        logger.info("Server shutdown complete.")

    app = FastAPI(title="Z-Image Hybrid API", lifespan=lifespan)
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # === API Endpoints ===

    @app.post("/session/create", response_model=CreateSessionResponse)
    async def create_session():
        meta = session_mgr.create()
        return {"session_id": meta.session_id, "available_styles": prompt_compiler.available_styles()}

    @app.get("/sessions")
    async def list_sessions():
        sessions = session_mgr.list_all()
        result = []
        for s in sessions:
            if s.total_versions > 0:
                result.append({
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "style_preset": s.style_preset,
                    "current_version": s.current_version,
                    "total_versions": s.total_versions,
                    "thumbnail_url": f"/session/{s.session_id}/thumbnail/{s.current_version}",
                })
        return result

    @app.post("/session/{session_id}/generate", response_model=GenerateResponse)
    async def generate(session_id: str, req: GenerateRequestModel):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        bundle = prompt_compiler.compile_poetry(
            poem_text=req.scene_text,
            style_preset=req.style_preset,
        )

        session_mgr.set_global_prompt(session_id, bundle.global_prompt, req.style_preset)

        gen_req = GenerateRequest(
            prompt=bundle.final_prompt,
            negative_prompt=bundle.negative_prompt,
            seed=req.seed,
            steps=req.steps,
            cfg=req.cfg,
            height=req.height,
            width=req.width,
        )
        result = engine.generate(gen_req)

        card = bundle.meta.copy()
        card["seed"] = result.seed
        card["final_prompt"] = bundle.final_prompt
        card["negative_prompt"] = bundle.negative_prompt
        card["engine"] = "Z-Image-Turbo"

        ver = session_mgr.save_version(
            session_id, result.image, card, edit_type="generate", edit_text=req.scene_text
        )

        return {
            "version": ver.version,
            "image_url": f"/session/{session_id}/image/{ver.version}",
            "prompt_card": card,
        }

    @app.post("/session/{session_id}/edit", response_model=EditResponse)
    async def edit(
        session_id: str,
        mask: UploadFile = File(...),
        edit_text: str = Form(...),
        seed: int = Form(-1),
        steps: int = Form(30),
        cfg: float = Form(7.5),
        strength: float = Form(0.6),
        grow_pixels: int = Form(8),
        blur_sigma: float = Form(12.0),
        invert_mask: bool = Form(False),
    ):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        current_img = session_mgr.current_image(session_id)
        if current_img is None:
            raise HTTPException(status_code=400, detail="No image to edit")

        mask_bytes = await mask.read()
        raw_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")

        processed_mask = mask_processor.process(
            raw_mask,
            target_size=current_img.size,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            invert=invert_mask,
        )

        meta = session_mgr.get_meta(session_id)
        bundle = prompt_compiler.compile_edit(
            edit_text=edit_text,
            global_style=meta.style_preset or "水墨",
        )

        inpaint_req = InpaintRequest(
            prompt=bundle.final_prompt,
            negative_prompt=bundle.negative_prompt,
            image=current_img,
            mask=processed_mask,
            seed=seed,
            steps=steps,
            cfg=cfg,
            strength=strength,
        )
        result = engine.inpaint(inpaint_req)

        card = bundle.meta.copy()
        card["seed"] = result.seed
        card["final_prompt"] = bundle.final_prompt
        card["negative_prompt"] = bundle.negative_prompt
        card["strength"] = strength
        card["engine"] = "SDXL-Inpaint"

        ver = session_mgr.save_version(
            session_id, result.image, card,
            edit_type="edit", edit_text=edit_text, mask=processed_mask
        )

        return {
            "version": ver.version,
            "image_url": f"/session/{session_id}/image/{ver.version}",
            "prompt_card": card,
        }

    @app.get("/session/{session_id}/history")
    async def history(session_id: str):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        items = session_mgr.history(session_id)
        return [
            {
                "version": it.version,
                "timestamp": it.timestamp,
                "edit_type": it.edit_type,
                "edit_text": it.edit_text,
            }
            for it in items
        ]

    @app.post("/session/{session_id}/revert")
    async def revert(session_id: str, version: int = Form(...)):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_mgr.revert(session_id, version)
        return {"status": "ok", "current_version": version}

    @app.get("/session/{session_id}/image/{version}")
    async def image_file(session_id: str, version: int):
        path = session_mgr.image_path(session_id, version)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/session/{session_id}/thumbnail/{version}")
    async def thumbnail_file(session_id: str, version: int):
        path = session_mgr.thumb_path(session_id, version)
        if not path.exists():
            path = session_mgr.image_path(session_id, version)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(path, media_type="image/png")

    return app


def main():
    parser = argparse.ArgumentParser(description="Z-Image Hybrid Server")
    parser.add_argument(
        "--zimage-model",
        type=str,
        default="E:\AIGC\models\Z-Image-Turbo-FP8",
        help="Path or HF model ID (default: E:\AIGC\models\Z-Image-Turbo-FP8)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp8", "bf16", "fp16", "fp32"],
        default="fp8",
        help="Model precision: fp8 (需要RTX40系), bf16, fp16, fp32 (default: fp8)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn
    app = create_app(args.zimage_model, args.dtype)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
