"""
Z-Image-Turbo-SDNQ + SDXL Inpaint Hybrid Engine
文生图使用 Z-Image-Turbo-SDNQ-int8 (INT8 量化)，编辑/Inpaint 使用 SDXL
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
    - text2img: Z-Image-Turbo-SDNQ-int8 (INT8 量化，显存友好)
    - inpaint: SDXL Inpainting (保持兼容性)
    """

    def __init__(
        self,
        zimage_path: str,
        inpaint_path: str,
        device: Literal["cuda", "cpu"] = "cuda",
        enable_cpu_offload: bool = True,
        use_torch_compile: bool = True,
    ):
        import torch

        self._torch = torch
        self.zimage_path = zimage_path
        self.inpaint_path = inpaint_path

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.torch_dtype = torch.bfloat16  # SDNQ 使用 bfloat16

        self.enable_cpu_offload = enable_cpu_offload
        self.use_torch_compile = use_torch_compile

        self._pipe_zimage = None
        self._pipe_inpaint = None

    def unload(self) -> None:
        self._pipe_zimage = None
        self._pipe_inpaint = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def generate(self, req: GenerateRequest) -> EngineResult:
        """使用 Z-Image-Turbo-SDNQ-int8 生成图像"""
        self._ensure_zimage()
        generator, seed = self._make_generator(req.seed)

        # Z-Image-Turbo 推荐参数: 8-9 步, cfg 0.0 (classifier-free guidance off)
        steps = req.steps if req.steps > 0 else 9
        cfg = req.cfg if req.cfg > 0 else 0.0  # SDNQ 版本推荐 0.0

        logger.info("Z-Image-SDNQ generate: %sx%s steps=%s cfg=%s seed=%s", 
                    req.width, req.height, steps, cfg, seed)
        
        out = self._pipe_zimage(
            prompt=req.prompt,
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
        if seed is None or seed < 0:
            seed = torch.seed()
        return torch.manual_seed(int(seed)), int(seed)

    def _ensure_zimage(self) -> None:
        if self._pipe_zimage is not None:
            return

        import torch
        import diffusers
        from sdnq import SDNQConfig  # 注册 SDNQ 到 diffusers
        from sdnq.common import use_torch_compile as triton_is_available
        from sdnq.loader import apply_sdnq_options_to_model

        logger.info("Loading Z-Image-Turbo-SDNQ-int8 from %s", self.zimage_path)
        
        # 加载 ZImagePipeline
        self._pipe_zimage = diffusers.ZImagePipeline.from_pretrained(
            self.zimage_path,
            torch_dtype=torch.bfloat16,
        )
        
        # 启用 INT8 MatMul (GPU 加速)
        if triton_is_available and torch.cuda.is_available():
            logger.info("Enabling INT8 quantized matmul for transformer and text_encoder")
            self._pipe_zimage.transformer = apply_sdnq_options_to_model(
                self._pipe_zimage.transformer, use_quantized_matmul=True
            )
            self._pipe_zimage.text_encoder = apply_sdnq_options_to_model(
                self._pipe_zimage.text_encoder, use_quantized_matmul=True
            )
            
            # 可选: torch.compile 加速
            if self.use_torch_compile:
                logger.info("Applying torch.compile to transformer for faster inference")
                self._pipe_zimage.transformer = torch.compile(self._pipe_zimage.transformer)
        
        # 启用 CPU offload 节省显存
        if self.enable_cpu_offload:
            self._pipe_zimage.enable_model_cpu_offload()
            logger.info("CPU offload enabled for Z-Image pipeline")

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
