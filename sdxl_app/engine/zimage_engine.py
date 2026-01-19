"""
Z-Image-Turbo + SDXL Inpaint Hybrid Engine
文生图使用 Z-Image-Turbo (DiT)，编辑/Inpaint 使用 SDXL
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Optional, Literal

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    height: int
    width: int


@dataclass(frozen=True)
class InpaintRequest:
    prompt: str
    negative_prompt: str
    image: Image.Image
    mask: Image.Image
    seed: int
    steps: int
    cfg: float
    strength: float


@dataclass(frozen=True)
class EngineResult:
    image: Image.Image
    seed: int


class ZImageHybridEngine:
    """
    混合推理引擎：
    - text2img: Z-Image-Turbo (DiT, 8步极速生成)
    - inpaint: SDXL Inpainting (保持兼容性)
    """

    def __init__(
        self,
        zimage_path: str,
        inpaint_path: str,
        device: Literal["cuda", "cpu"] = "cuda",
        dtype: Literal["bf16", "fp16", "fp32", "fp8"] = "bf16",
        enable_cpu_offload: bool = True,
    ):
        import torch

        self._torch = torch
        self.zimage_path = zimage_path
        self.inpaint_path = inpaint_path

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.dtype = dtype
        
        # 设置 torch dtype
        if dtype == "fp8":
            # FP8 需要 PyTorch 2.1+ 和 CUDA 12+
            if hasattr(torch, 'float8_e4m3fn'):
                self.torch_dtype = torch.float8_e4m3fn
                logger.info("Using FP8 (float8_e4m3fn) precision")
            else:
                logger.warning("FP8 not supported, falling back to BF16")
                self.torch_dtype = torch.bfloat16
        elif dtype == "bf16":
            self.torch_dtype = torch.bfloat16
        elif dtype == "fp16":
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        self.enable_cpu_offload = enable_cpu_offload

        self._pipe_zimage = None
        self._pipe_inpaint = None

    def unload(self) -> None:
        self._pipe_zimage = None
        self._pipe_inpaint = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def generate(self, req: GenerateRequest) -> EngineResult:
        """使用 Z-Image-Turbo 生成图像"""
        self._ensure_zimage()
        generator, seed = self._make_generator(req.seed)

        # Z-Image-Turbo 推荐参数: 8 步, cfg 3.5
        steps = min(req.steps, 8) if req.steps <= 8 else req.steps
        cfg = req.cfg if req.cfg > 0 else 3.5

        logger.info("Z-Image generate: %sx%s steps=%s cfg=%s seed=%s", 
                    req.width, req.height, steps, cfg, seed)
        
        out = self._pipe_zimage(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt if req.negative_prompt else None,
            height=req.height,
            width=req.width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        ).images[0]

        return EngineResult(image=out, seed=seed)

    def inpaint(self, req: InpaintRequest) -> EngineResult:
        """使用 SDXL Inpainting 编辑图像"""
        self._ensure_inpaint()
        generator, seed = self._make_generator(req.seed)

        base = req.image.convert("RGB")
        mask = req.mask.convert("L")

        logger.info("SDXL inpaint: strength=%s steps=%s cfg=%s seed=%s", 
                    req.strength, req.steps, req.cfg, seed)
        
        out = self._pipe_inpaint(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=base,
            mask_image=mask,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            strength=req.strength,
            generator=generator,
        ).images[0]

        return EngineResult(image=out, seed=seed)

    def _make_generator(self, seed: int):
        torch = self._torch
        gen = torch.Generator(device=self.device)
        if seed is None or seed < 0:
            seed = gen.seed()
        gen.manual_seed(int(seed))
        return gen, int(seed)

    def _ensure_zimage(self) -> None:
        if self._pipe_zimage is not None:
            return

        from diffusers import DiffusionPipeline

        logger.info("Loading Z-Image-Turbo pipeline from %s (dtype=%s)", self.zimage_path, self.dtype)
        
        # DiffusionPipeline 自动识别模型类型
        # 支持本地路径或 HuggingFace 模型 ID (如 "T5B/Z-Image-Turbo-FP8")
        self._pipe_zimage = DiffusionPipeline.from_pretrained(
            self.zimage_path,
            torch_dtype=self.torch_dtype,
            device_map="balanced" if self.device == "cuda" else None,
        )
        
        if self.enable_cpu_offload and self.device == "cuda":
            try:
                self._pipe_zimage.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("CPU offload not enabled: %s", e)

    def _ensure_inpaint(self) -> None:
        if self._pipe_inpaint is not None:
            return

        from diffusers import AutoPipelineForInpainting

        logger.info("Loading SDXL inpaint pipeline from %s", self.inpaint_path)
        self._pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_path,
            torch_dtype=self._torch.float16,  # SDXL 用 fp16
            use_safetensors=True,
            variant="fp16",
        )
        
        if self.enable_cpu_offload:
            try:
                self._pipe_inpaint.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("CPU offload not enabled: %s", e)
                self._pipe_inpaint.to(self.device)
        else:
            self._pipe_inpaint.to(self.device)
