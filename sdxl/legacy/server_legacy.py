# NOTE: legacy entrypoint kept for reference; new API is `sdxl_app.api.server`.
# ============================================================
# SDXL Inpainting Editor - FastAPI Backend
# ============================================================
"""
FastAPI 后端服务：
- 提供 Session 管理接口
- 提供图像生成与编辑接口
- 提供历史记录与回退接口
"""
import logging
import io
import base64
from typing import Optional, List
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, Response, FileResponse
from pydantic import BaseModel
from PIL import Image
import uvicorn

from config import API_HOST, API_PORT, DEFAULT_GENERATE_PARAMS, DEFAULT_EDIT_PARAMS
from engine.sdxl_inpaint import get_engine
from engine.mask_utils import preprocess_mask, alpha_blend
from engine.prompt_utils import (
    build_generation_prompt, 
    build_edit_prompt, 
    create_generation_card,
    create_edit_card,
    get_available_styles
)
from storage.session_manager import get_session_manager, VersionInfo

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# 全局管理器
session_manager = get_session_manager()
engine = get_engine()

# ============================================================
# 生命周期管理
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动加载模型，关闭卸载模型"""
    logger.info("Server starting...")
    # 可以在这里预加载模型，或者让 engine 懒加载
    yield
    logger.info("Server shutting down...")
    engine.unload_models()

app = FastAPI(title="SDXL Inpainting API", lifespan=lifespan)


# ============================================================
# 数据模型
# ============================================================
class CreateSessionResponse(BaseModel):
    session_id: str
    available_styles: List[str]

class GenerateRequest(BaseModel):
    style_preset: str
    scene_text: str
    strength: Optional[float] = None  # 用于 img2img refiner（暂未用）
    # 参数覆盖（可选）
    seed: int = -1
    steps: int = DEFAULT_GENERATE_PARAMS["steps"]
    cfg: float = DEFAULT_GENERATE_PARAMS["cfg"]
    height: int = DEFAULT_GENERATE_PARAMS["height"]
    width: int = DEFAULT_GENERATE_PARAMS["width"]

class GenerateResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict

class EditRequest(BaseModel):
    edit_text: str
    # 参数覆盖（可选）
    seed: int = -1
    steps: int = DEFAULT_EDIT_PARAMS["steps"]
    cfg: float = DEFAULT_EDIT_PARAMS["cfg"]
    strength: float = DEFAULT_EDIT_PARAMS["strength"]
    grow_pixels: int = DEFAULT_EDIT_PARAMS["grow_pixels"]
    blur_sigma: float = DEFAULT_EDIT_PARAMS["blur_sigma"]
    invert_mask: bool = DEFAULT_EDIT_PARAMS["invert_mask"]

class EditResponse(BaseModel):
    version: int
    image_url: str
    prompt_card: dict

class SimpleResponse(BaseModel):
    success: bool
    message: str


# ============================================================
# 辅助函数
# ============================================================
def pil_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def decode_base64_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data))


# ============================================================
# API 端点
# ============================================================

@app.post("/session/create", response_model=CreateSessionResponse)
async def create_session():
    """创建新会话"""
    session_id = session_manager.create_session()
    return {
        "session_id": session_id,
        "available_styles": get_available_styles()
    }

@app.get("/session/{session_id}/check")
async def check_session(session_id: str):
    """检查会话是否存在"""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"exists": True}

@app.post("/session/{session_id}/generate", response_model=GenerateResponse)
async def generate_v0(session_id: str, req: GenerateRequest):
    """文生图：生成 v0 版本"""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # 1. 构建 Prompt
        global_prompt, final_prompt, negative_prompt = build_generation_prompt(
            style=req.style_preset,
            scene_text=req.scene_text
        )
        
        # 2. 只有生成成功才设置 global_prompt
        # 调用推理
        image, actual_seed = engine.generate(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=req.seed,
            steps=req.steps,
            cfg=req.cfg,
            height=req.height,
            width=req.width
        )
        
        # 3. 保存 session 全局设置
        session_manager.set_global_prompt(session_id, global_prompt, req.style_preset)
        
        # 4. 创建 PromptCard
        card = create_generation_card(
            style=req.style_preset,
            scene_text=req.scene_text,
            global_prompt=global_prompt,
            final_prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=actual_seed,
            steps=req.steps,
            cfg=req.cfg,
            height=req.height,
            width=req.width
        )
        
        # 5. 保存版本
        version = session_manager.save_version(
            session_id=session_id,
            image=image,
            params=card.to_dict(),
            edit_type="generate",
            edit_text=req.scene_text
        )
        
        return {
            "version": version,
            "image_url": f"/session/{session_id}/image/{version}",
            "prompt_card": card.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Generate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/{session_id}/edit", response_model=EditResponse)
async def edit_image(
    session_id: str,
    mask: UploadFile = File(...),
    edit_text: str = Form(...),
    # 参数
    seed: int = Form(-1),
    steps: int = Form(DEFAULT_EDIT_PARAMS["steps"]),
    cfg: float = Form(DEFAULT_EDIT_PARAMS["cfg"]),
    strength: float = Form(DEFAULT_EDIT_PARAMS["strength"]),
    grow_pixels: int = Form(DEFAULT_EDIT_PARAMS["grow_pixels"]),
    blur_sigma: float = Form(DEFAULT_EDIT_PARAMS["blur_sigma"]),
    invert_mask: bool = Form(DEFAULT_EDIT_PARAMS["invert_mask"])
):
    """Inpainting 编辑"""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # 1. 获取输入数据
        current_image = session_manager.get_current_image(session_id)
        if current_image is None:
            raise HTTPException(status_code=400, detail="Cannot edit before generating v0")
        
        mask_bytes = await mask.read()
        mask_image = Image.open(io.BytesIO(mask_bytes))
        
        metadata = session_manager.get_metadata(session_id)
        global_prompt = metadata.global_prompt or ""
        
        # 2. Mask 预处理
        processed_mask, alpha_mask = preprocess_mask(
            mask=mask_image,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            invert=invert_mask
        )
        
        # 3. 构建 Prompt
        final_prompt, negative_prompt = build_edit_prompt(
            global_prompt=global_prompt,
            edit_text=edit_text
        )
        
        # 4. Inpainting 推理
        edited, actual_seed = engine.inpaint(
            image=current_image,
            mask=processed_mask,
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            strength=strength
        )
        
        # 5. 后处理融合 (Alpha Blend)
        final_image = alpha_blend(
            edited=edited,
            original=current_image,
            alpha_mask=alpha_mask
        )
        
        # 6. 创建 PromptCard
        card = create_edit_card(
            global_prompt=global_prompt,
            edit_text=edit_text,
            final_prompt=final_prompt,
            negative_prompt=negative_prompt,
            seed=actual_seed,
            steps=steps,
            cfg=cfg,
            strength=strength,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            style_preset=metadata.style_preset
        )
        
        # 7. 保存版本
        # 保存原始 mask（用于记录用户输入）
        version = session_manager.save_version(
            session_id=session_id,
            image=final_image,
            params=card.to_dict(),
            edit_type="edit",
            edit_text=edit_text,
            mask=mask_image
        )
        
        return {
            "version": version,
            "image_url": f"/session/{session_id}/image/{version}",
            "prompt_card": card.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Edit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}/history")
async def get_history(session_id: str):
    """获取历史版本列表"""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = session_manager.get_history(session_id)
    # 将 thumbnail_path 转换为 URL
    result = []
    for h in history:
        h_dict = asdict(h)
        h_dict["thumbnail_url"] = f"/session/{session_id}/thumbnail/{h.version}"
        h_dict["image_url"] = f"/session/{session_id}/image/{h.version}"
        del h_dict["thumbnail_path"]
        result.append(h_dict)
    
    return result

@app.post("/session/{session_id}/revert")
async def revert_version(session_id: str, version: int = Form(...)):
    """回退到指定版本"""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_manager.revert_to_version(session_id, version)
        return {"success": True, "message": f"Reverted to version {version}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/session/{session_id}/image/{version}")
async def get_image_file(session_id: str, version: int):
    """获取指定版本的图片"""
    try:
        # 直接使用文件路径响应，最高效
        session_dir = session_manager._get_session_dir(session_id)
        image_path = session_dir / f"v{version}.png"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(image_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/session/{session_id}/thumbnail/{version}")
async def get_thumbnail_file(session_id: str, version: int):
    """获取指定版本的缩略图"""
    try:
        session_dir = session_manager._get_session_dir(session_id)
        thumb_path = session_dir / f"v{version}_thumb.png"
        if not thumb_path.exists():
             # 如果缩略图不存在，回退到原图
            thumb_path = session_dir / f"v{version}.png"
            
        if not thumb_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return FileResponse(thumb_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
