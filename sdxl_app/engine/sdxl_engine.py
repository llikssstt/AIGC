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
    use_refiner: bool = False


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
    use_refiner: bool = False


@dataclass(frozen=True)
class EngineResult:
    image: Image.Image
    seed: int


class SDXLEngine:
    """
    统一推理引擎：text2img + inpaint，可选 refiner。
    """

    def __init__(
        self,
        base_path: str,
        inpaint_path: str,
        refiner_path: Optional[str],
        device: Literal["cuda", "cpu"] = "cuda",
        dtype: Literal["fp16", "fp32"] = "fp16",
        enable_xformers: bool = True,
        enable_cpu_offload: bool = True,
        enable_vae_slicing: bool = False,
        enable_vae_tiling: bool = False,
    ):
        import torch

        self._torch = torch
        self.base_path = base_path
        self.inpaint_path = inpaint_path
        self.refiner_path = refiner_path

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.dtype = dtype
        self.torch_dtype = torch.float16 if (dtype == "fp16" and self.device == "cuda") else torch.float32

        self.enable_xformers = enable_xformers
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_slicing = enable_vae_slicing
        self.enable_vae_tiling = enable_vae_tiling

        self._pipe_text2img = None
        self._pipe_inpaint = None
        self._pipe_refiner = None

    def unload(self) -> None:
        self._pipe_text2img = None
        self._pipe_inpaint = None
        self._pipe_refiner = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        gc.collect()

    def generate(self, req: GenerateRequest) -> EngineResult:
        self._ensure_text2img()
        generator, seed = self._make_generator(req.seed)

        logger.info("text2img: %sx%s steps=%s cfg=%s seed=%s", req.width, req.height, req.steps, req.cfg, seed)
        out = self._pipe_text2img(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            generator=generator,
        ).images[0]

        if req.use_refiner and self.refiner_path:
            out = self._refine(out, req.prompt, req.negative_prompt, generator=generator)

        return EngineResult(image=out, seed=seed)

    def inpaint(self, req: InpaintRequest) -> EngineResult:
        self._ensure_inpaint()
        generator, seed = self._make_generator(req.seed)

        base = req.image.convert("RGB")
        mask = req.mask.convert("L")

        logger.info("inpaint: strength=%s steps=%s cfg=%s seed=%s", req.strength, req.steps, req.cfg, seed)
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

        if req.use_refiner and self.refiner_path:
            out = self._refine(out, req.prompt, req.negative_prompt, generator=generator)

        return EngineResult(image=out, seed=seed)

    def _make_generator(self, seed: int):
        torch = self._torch
        gen = torch.Generator(device=self.device)
        if seed is None or seed < 0:
            seed = gen.seed()
        gen.manual_seed(int(seed))
        return gen, int(seed)

    def _configure_pipe(self, pipe) -> None:
        if self.device == "cuda":
            try:
                if self.enable_xformers:
                    pipe.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning("xformers not enabled: %s", e)

            if self.enable_vae_slicing:
                try:
                    pipe.enable_vae_slicing()
                except Exception:
                    pass
            if self.enable_vae_tiling:
                try:
                    pipe.enable_vae_tiling()
                except Exception:
                    pass

            if self.enable_cpu_offload:
                try:
                    pipe.enable_model_cpu_offload()
                    return
                except Exception as e:
                    logger.warning("cpu offload not enabled: %s", e)

        pipe.to(self.device)

    def _ensure_text2img(self) -> None:
        if self._pipe_text2img is not None:
            return

        from diffusers import AutoPipelineForText2Image

        logger.info("Loading text2img pipeline from %s", self.base_path)
        self._pipe_text2img = AutoPipelineForText2Image.from_pretrained(
            self.base_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16" if self.torch_dtype == self._torch.float16 else None,
        )
        self._configure_pipe(self._pipe_text2img)

    def _ensure_inpaint(self) -> None:
        if self._pipe_inpaint is not None:
            return

        from diffusers import AutoPipelineForInpainting

        logger.info("Loading inpaint pipeline from %s", self.inpaint_path)
        self._pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
            self.inpaint_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16" if self.torch_dtype == self._torch.float16 else None,
        )
        self._configure_pipe(self._pipe_inpaint)

    def _ensure_refiner(self) -> None:
        if self._pipe_refiner is not None:
            return
        if not self.refiner_path:
            return

        from diffusers import StableDiffusionXLImg2ImgPipeline

        logger.info("Loading refiner pipeline from %s", self.refiner_path)
        self._pipe_refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.refiner_path,
            torch_dtype=self.torch_dtype,
            use_safetensors=True,
            variant="fp16" if self.torch_dtype == self._torch.float16 else None,
        )
        self._configure_pipe(self._pipe_refiner)

    def _refine(self, image: Image.Image, prompt: str, negative_prompt: str, generator) -> Image.Image:
        self._ensure_refiner()
        if self._pipe_refiner is None:
            return image

        out = self._pipe_refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=0.25,
            num_inference_steps=20,
            guidance_scale=5.0,
            generator=generator,
        ).images[0]
        return out

