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
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_fuse: bool = True,
        device: Literal["cuda", "cpu"] = "cuda",
        enable_cpu_offload: bool = True,
        use_torch_compile: bool = True,
    ):
        import torch

        self._torch = torch
        self.zimage_path = zimage_path
        self.inpaint_path = inpaint_path
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.lora_fuse = lora_fuse

        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.torch_dtype = torch.bfloat16  # SDNQ 使用 bfloat16

        self.enable_cpu_offload = enable_cpu_offload
        self.use_torch_compile = use_torch_compile

        self._pipe_zimage = None
        self._pipe_inpaint = None
        self._lora_cross_attention_scale: Optional[float] = None

    def _maybe_apply_lora(self, pipe) -> None:
        if not self.lora_path:
            return

        lora_path = self.lora_path
        scale = float(self.lora_scale)
        errors: list[str] = []
        loaded = False
        needs_prefix_none = False
        legacy_attn_procs = False
        weight_name: Optional[str] = None

        try:
            from pathlib import Path

            p = Path(lora_path)
            if p.is_dir():
                candidate = p / "pytorch_lora_weights.safetensors"
                if candidate.exists():
                    weight_name = candidate.name
                else:
                    first = next(iter(sorted(p.glob("*.safetensors"))), None)
                    if first is not None:
                        weight_name = first.name
            elif p.is_file():
                weight_name = p.name

            if weight_name:
                from safetensors import safe_open  # type: ignore

                weight_file = p / weight_name if p.is_dir() else p
                with safe_open(str(weight_file), framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                legacy_attn_procs = any(
                    k.startswith(("down_blocks.", "up_blocks.", "mid_block.")) and (".lora_A." in k or ".lora_B." in k)
                    for k in keys
                )
                has_known_prefix = any(k.startswith(("unet.", "text_encoder", "text_encoder_2")) for k in keys)
                needs_prefix_none = (not has_known_prefix) and (not legacy_attn_procs)
        except Exception as e:
            logger.debug("LoRA format inspection skipped: %s", e)

        logger.info("Loading LoRA for SDXL inpaint from %s (scale=%s)", lora_path, scale)

        if legacy_attn_procs:
            logger.info("Detected legacy LoRA attention-processor weights; using unet.load_attn_procs")
            unet = getattr(pipe, "unet", None)
            unet_loader = getattr(unet, "load_attn_procs", None)
            if callable(unet_loader):
                try:
                    from pathlib import Path

                    p = Path(lora_path)
                    lora_dir = p if p.is_dir() else p.parent
                    try:
                        unet_loader(str(lora_dir), weight_name=weight_name)
                    except TypeError:
                        unet_loader(str(lora_dir))
                    loaded = True
                except Exception as e:
                    errors.append(repr(e))
            else:
                errors.append("legacy attn-procs LoRA detected but unet.load_attn_procs is unavailable")

        loader = getattr(pipe, "load_lora_weights", None)
        if not loaded and callable(loader):
            try:
                if needs_prefix_none:
                    loader(lora_path, adapter_name="default", prefix=None, weight_name=weight_name)
                else:
                    loader(lora_path, adapter_name="default", weight_name=weight_name)
                loaded = True
            except TypeError:
                try:
                    if needs_prefix_none:
                        loader(lora_path, prefix=None, weight_name=weight_name)
                    else:
                        loader(lora_path, weight_name=weight_name)
                    loaded = True
                except Exception as e:
                    errors.append(repr(e))
            except Exception as e:
                errors.append(repr(e))

        if not loaded and weight_name and callable(loader) and (needs_prefix_none or legacy_attn_procs):
            try:
                from pathlib import Path
                from safetensors.torch import load_file  # type: ignore

                p = Path(lora_path)
                weight_file = p / weight_name if p.is_dir() else p
                state = load_file(str(weight_file))
                state = {f"unet.{k}": v for k, v in state.items()}
                loader(state, adapter_name="default")
                loaded = True
                errors = []
            except Exception as e:
                errors.append(repr(e))

        if not loaded and not legacy_attn_procs:
            unet = getattr(pipe, "unet", None)
            unet_loader = getattr(unet, "load_attn_procs", None)
            if callable(unet_loader):
                try:
                    unet_loader(lora_path)
                    loaded = True
                except Exception as e:
                    errors.append(repr(e))

        if not loaded:
            msg = "; ".join(errors) if errors else "unknown error"
            raise RuntimeError(f"Failed to load LoRA from '{lora_path}': {msg}")

        self._lora_cross_attention_scale = None
        if legacy_attn_procs:
            self._lora_cross_attention_scale = scale
        elif self.lora_fuse:
            fuse = getattr(pipe, "fuse_lora", None)
            if callable(fuse):
                try:
                    fuse(lora_scale=scale)
                except TypeError:
                    fuse(scale)
            else:
                self._lora_cross_attention_scale = scale
        else:
            self._lora_cross_attention_scale = scale

    def _cross_attention_kwargs(self) -> Optional[dict]:
        if self._lora_cross_attention_scale is None:
            return None
        return {"scale": float(self._lora_cross_attention_scale)}

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

        extra = {}
        cross_attention_kwargs = self._cross_attention_kwargs()
        if cross_attention_kwargs is not None:
            extra["cross_attention_kwargs"] = cross_attention_kwargs

        out = self._pipe_inpaint(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            image=base,
            mask_image=mask,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            strength=req.strength,
            generator=generator,
            **extra,
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
        self._maybe_apply_lora(self._pipe_inpaint)

        if self.enable_cpu_offload:
            try:
                self._pipe_inpaint.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("CPU offload not enabled: %s", e)
                self._pipe_inpaint.to(self.device)
        else:
            self._pipe_inpaint.to(self.device)
