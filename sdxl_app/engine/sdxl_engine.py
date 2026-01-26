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
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_fuse: bool = True,
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
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.lora_fuse = lora_fuse

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
        self._lora_cross_attention_scale: Optional[float] = None

    def _maybe_apply_lora(self, pipe, purpose: str) -> None:
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

        logger.info("Loading LoRA for %s from %s (scale=%s)", purpose, lora_path, scale)

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
        extra = {}
        cross_attention_kwargs = self._cross_attention_kwargs()
        if cross_attention_kwargs is not None:
            extra["cross_attention_kwargs"] = cross_attention_kwargs

        out = self._pipe_text2img(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            generator=generator,
            **extra,
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
        self._maybe_apply_lora(self._pipe_text2img, purpose="text2img")
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
        self._maybe_apply_lora(self._pipe_inpaint, purpose="inpaint")
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

