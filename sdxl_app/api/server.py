from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from sdxl_app.config import get_settings
from sdxl_app.engine.sdxl_engine import SDXLEngine, GenerateRequest, InpaintRequest
from sdxl_app.engine.prompt_utils import PromptCompiler
from sdxl_app.engine.mask_utils import MaskProcessor
from sdxl_app.engine.llm_service import get_llm_service
from sdxl_app.storage.session_store import SessionManager

logger = logging.getLogger(__name__)


class CreateSessionResponse(BaseModel):
    session_id: str
    available_styles: list[str]


class GenerateRequestModel(BaseModel):
    style_preset: str
    scene_text: str
    seed: int = -1
    steps: int = 30
    cfg: float = 7.5
    height: int = 1024
    width: int = 1024
    use_refiner: bool = False


class GenerateResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict


class EditResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict


class ImportResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict


def create_app() -> FastAPI:
    settings = get_settings()

    session_mgr = SessionManager(settings.storage.sessions_dir)
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
        poetry_preamble=settings.prompts.poetry_preamble,
        poetry_negative_append=settings.prompts.poetry_negative_append,
        llm_service=llm_service,
    )
    mask_processor = MaskProcessor()
    engine = SDXLEngine(
        base_path=settings.models.base_path,
        inpaint_path=settings.models.inpaint_path,
        refiner_path=settings.models.refiner_path,
        device=settings.runtime.device,
        dtype=settings.runtime.dtype,
        enable_xformers=settings.runtime.enable_xformers,
        enable_cpu_offload=settings.runtime.enable_cpu_offload,
        enable_vae_slicing=settings.runtime.enable_vae_slicing,
        enable_vae_tiling=settings.runtime.enable_vae_tiling,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine.unload()

    app = FastAPI(title="SDXL Local API", lifespan=lifespan)

    @app.post("/session/create", response_model=CreateSessionResponse)
    async def create_session():
        meta = session_mgr.create()
        return {"session_id": meta.session_id, "available_styles": prompt_compiler.available_styles()}

    @app.post("/session/{session_id}/generate", response_model=GenerateResponse)
    async def generate(session_id: str, req: GenerateRequestModel):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        bundle = prompt_compiler.compile_generation(req.style_preset, req.scene_text)
        result = engine.generate(
            GenerateRequest(
                prompt=bundle.final_prompt,
                negative_prompt=bundle.negative_prompt,
                seed=req.seed,
                steps=req.steps,
                cfg=req.cfg,
                height=req.height,
                width=req.width,
                use_refiner=req.use_refiner,
            )
        )

        card = prompt_compiler.generation_card(
            style_preset=req.style_preset,
            scene_text=req.scene_text,
            bundle=bundle,
            seed=result.seed,
            steps=req.steps,
            cfg=req.cfg,
            width=req.width,
            height=req.height,
        )
        session_mgr.set_global_prompt(session_id, bundle.global_prompt, req.style_preset)
        v = session_mgr.save_version(session_id, result.image, card, edit_type="generate", edit_text=req.scene_text)

        return {
            "version": v.version,
            "image_url": f"/session/{session_id}/image/{v.version}",
            "prompt_card": card,
        }

    @app.post("/session/{session_id}/import", response_model=ImportResponse)
    async def import_image(session_id: str, image: UploadFile = File(...)):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        raw = await image.read()
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

        # import 不改变 global_prompt/style_preset（用户可后续 generate 一次设定风格）
        card = prompt_compiler.import_card()
        v = session_mgr.save_version(session_id, pil, card, edit_type="import", edit_text=None, mask=None)

        return {
            "version": v.version,
            "image_url": f"/session/{session_id}/image/{v.version}",
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
        use_refiner: bool = Form(False),
    ):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        base_img = session_mgr.current_image(session_id)
        if base_img is None:
            raise HTTPException(status_code=400, detail="No base image in session")

        mask_bytes = await mask.read()
        try:
            raw_mask = Image.open(io.BytesIO(mask_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid mask: {e}")

        processed = mask_processor.preprocess(
            raw_mask,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            invert=invert_mask,
        )

        meta = session_mgr.get_meta(session_id)
        bundle = prompt_compiler.compile_edit(meta.global_prompt, edit_text)

        out = engine.inpaint(
            InpaintRequest(
                prompt=bundle.final_prompt,
                negative_prompt=bundle.negative_prompt,
                image=base_img,
                mask=processed.mask_for_inpaint,
                seed=seed,
                steps=steps,
                cfg=cfg,
                strength=strength,
                use_refiner=use_refiner,
            )
        )

        final_img = mask_processor.alpha_blend(out.image, base_img, processed.alpha)

        card = prompt_compiler.edit_card(
            style_preset=meta.style_preset,
            global_prompt=meta.global_prompt,
            edit_text=edit_text,
            bundle=bundle,
            seed=out.seed,
            steps=steps,
            cfg=cfg,
            strength=strength,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            invert_mask=invert_mask,
        )
        v = session_mgr.save_version(session_id, final_img, card, edit_type="edit", edit_text=edit_text, mask=raw_mask)

        return {
            "version": v.version,
            "image_url": f"/session/{session_id}/image/{v.version}",
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
                "edit_type": it.edit_type,
                "edit_text": it.edit_text,
                "image_url": f"/session/{session_id}/image/{it.version}",
                "thumbnail_url": f"/session/{session_id}/thumbnail/{it.version}",
                "prompt_card": it.card,
            }
            for it in items
        ]

    @app.post("/session/{session_id}/revert")
    async def revert(session_id: str, version: int = Form(...)):
        if not session_mgr.exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        session_mgr.revert(session_id, version)
        return {"success": True}

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
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
