from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from sdxl_app.config import Settings, configure_logging
from sdxl_app.engine.mask_utils import MaskProcessor
from sdxl_app.engine.sdxl_engine import SDXLEngine, GenerateRequest, InpaintRequest


logger = logging.getLogger(__name__)


def _load_settings(config_path: Optional[str]) -> Settings:
    settings = Settings.load(config_file=config_path) if config_path else Settings.load()
    configure_logging(settings)
    return settings


def _save_image(image: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    logger.info("Saved: %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick SDXL text2img / inpaint runner (local diffusers folders).")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path (same as SDXL_CONFIG).")

    sub = parser.add_subparsers(dest="mode", required=True)

    p_txt = sub.add_parser("text2img", help="Generate an image from text.")
    p_txt.add_argument("--prompt", type=str, required=True)
    p_txt.add_argument("--negative", type=str, default=None, help="Negative prompt (default from config).")
    p_txt.add_argument("--seed", type=int, default=-1)
    p_txt.add_argument("--steps", type=int, default=None)
    p_txt.add_argument("--cfg", type=float, default=None)
    p_txt.add_argument("--width", type=int, default=None)
    p_txt.add_argument("--height", type=int, default=None)
    p_txt.add_argument("--out", type=str, default="outputs/text2img.png")

    p_inp = sub.add_parser("inpaint", help="Inpaint an image with a mask.")
    p_inp.add_argument("--prompt", type=str, required=True)
    p_inp.add_argument("--negative", type=str, default=None, help="Negative prompt (default from config).")
    p_inp.add_argument("--image", type=str, required=True, help="Base image path.")
    p_inp.add_argument("--mask", type=str, required=True, help="Mask path (white=edit, black=keep).")
    p_inp.add_argument("--seed", type=int, default=-1)
    p_inp.add_argument("--steps", type=int, default=None)
    p_inp.add_argument("--cfg", type=float, default=None)
    p_inp.add_argument("--strength", type=float, default=0.6)
    p_inp.add_argument("--mask-grow", type=int, default=8)
    p_inp.add_argument("--mask-blur", type=float, default=12.0)
    p_inp.add_argument("--invert-mask", action="store_true")
    p_inp.add_argument("--out", type=str, default="outputs/inpaint.png")

    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--base-model", type=str, default=None, help="SDXL base model folder (diffusers).")
    parser.add_argument("--inpaint-model", type=str, default=None, help="SDXL inpaint model folder (diffusers).")
    parser.add_argument("--refiner-model", type=str, default=None, help="Optional SDXL refiner folder (diffusers).")
    parser.add_argument("--no-xformers", action="store_true")
    parser.add_argument("--no-cpu-offload", action="store_true")

    args = parser.parse_args()
    settings = _load_settings(args.config)

    device = args.device or settings.runtime.device
    dtype = args.dtype or settings.runtime.dtype

    base_path = args.base_model or settings.models.base_path
    inpaint_path = args.inpaint_model or settings.models.inpaint_path
    refiner_path = args.refiner_model if args.refiner_model is not None else settings.models.refiner_path

    engine = SDXLEngine(
        base_path=base_path,
        inpaint_path=inpaint_path,
        refiner_path=refiner_path,
        device=device,
        dtype=dtype,
        enable_xformers=not args.no_xformers,
        enable_cpu_offload=not args.no_cpu_offload,
        enable_vae_slicing=settings.runtime.enable_vae_slicing,
        enable_vae_tiling=settings.runtime.enable_vae_tiling,
    )

    if args.mode == "text2img":
        steps = int(args.steps) if args.steps is not None else settings.defaults.generate_steps
        cfg = float(args.cfg) if args.cfg is not None else settings.defaults.generate_cfg
        width = int(args.width) if args.width is not None else settings.defaults.generate_width
        height = int(args.height) if args.height is not None else settings.defaults.generate_height
        negative = args.negative if args.negative is not None else settings.prompts.negative_prompt

        req = GenerateRequest(
            prompt=args.prompt,
            negative_prompt=negative,
            seed=int(args.seed),
            steps=steps,
            cfg=cfg,
            height=height,
            width=width,
            use_refiner=False,
        )
        result = engine.generate(req)
        _save_image(result.image, Path(args.out))
        return

    processor = MaskProcessor()
    base_img = Image.open(args.image).convert("RGB")
    mask_img = Image.open(args.mask)
    processed = processor.preprocess(
        mask=mask_img,
        grow_pixels=int(args.mask_grow),
        blur_sigma=float(args.mask_blur),
        invert=bool(args.invert_mask),
    )

    steps = int(args.steps) if args.steps is not None else settings.defaults.edit_steps
    cfg = float(args.cfg) if args.cfg is not None else settings.defaults.edit_cfg
    negative = args.negative if args.negative is not None else settings.prompts.negative_prompt

    req = InpaintRequest(
        prompt=args.prompt,
        negative_prompt=negative + settings.prompts.inpaint_negative_append,
        image=base_img,
        mask=processed.mask_for_inpaint,
        seed=int(args.seed),
        steps=steps,
        cfg=cfg,
        strength=float(args.strength),
        use_refiner=False,
    )
    result = engine.inpaint(req)
    blended = processor.alpha_blend(result.image, base_img, processed.alpha)
    _save_image(blended, Path(args.out))


if __name__ == "__main__":
    main()

