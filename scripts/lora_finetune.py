"""
Caption generation and LoRA fine-tuning workflow for the sorted Chinese landscape paintings.

Steps:

1. Ensure every image under `sorted_by_style/[水墨,工笔,青绿]` has a caption in `LoRA/captions.csv`.
   When the table is missing or a row is absent, the script prompts the local Qwen3 service to
   describe the work with the additional style guidance, then appends the caption to the table.
2. Build a simple dataset from the caption table and fine-tune LoRA adapters for both the UNet and the
   CLIP text encoder inside Stable Diffusion.
3. Save the LoRA adapters into subfolders under `LoRA/`.

Run `python scripts/lora_finetune.py --help` for full runtime options.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

STYLE_NAMES = ("水墨", "工笔", "青绿")
STYLE_GUIDANCE = (
    "水墨：以墨为主，强调留白、墨韵与气的流动；色彩有限，灰度/黑白对比突出。\n"
    "工笔：以精细描绘为核心，线条分明、结构精致，细节处有层层设色或罩染。\n"
    "青绿：以矿物颜料（石青/石绿）设色，颜色鲜明，可在工笔或写意框架下出现。"
)


@dataclass
class CaptionEntry:
    style: str
    relative_path: str
    caption: str
    last_updated: int

    def to_row(self) -> List[str]:
        return [
            self.style,
            self.relative_path,
            self.caption.replace("\n", " ").strip(),
            str(self.last_updated),
        ]


def read_caption_table(table_path: Path) -> Dict[str, CaptionEntry]:
    if not table_path.exists():
        return {}
    entries: Dict[str, CaptionEntry] = {}
    with table_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row.get("relative_path"):
                continue
            key = f"{row['style']}|{row['relative_path']}"
            entries[key] = CaptionEntry(
                style=row["style"],
                relative_path=row["relative_path"],
                caption=row["caption"],
                last_updated=int(row.get("last_updated", "0")),
            )
    return entries


def write_caption_table(table_path: Path, entries: Iterable[CaptionEntry]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["style", "relative_path", "caption", "last_updated"])
        for entry in entries:
            writer.writerow(entry.to_row())


def ensure_sdxl_weight_aliases(pretrained_path: Path, default_max_copy_bytes: int = 3 * 1024 * 1024 * 1024) -> None:
    """Create aliases for common SDXL weight filenames.

    Many offline SDXL folders store fp16 variant weights like:
    - text_encoder/model.fp16.safetensors
    - unet/diffusion_pytorch_model.fp16.safetensors

    Some diffusers/transformers combinations don't propagate `variant` to transformers submodels.
    Creating `model.safetensors` aliases avoids "no file named model.safetensors" errors.
    """

    if not pretrained_path.is_dir():
        return

    def ensure_weight_alias(source_rel: str, target_rel: str, max_copy_bytes: Optional[int] = None) -> None:
        source_path = pretrained_path / source_rel
        target_path = pretrained_path / target_rel
        if not source_path.exists() or target_path.exists():
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, target_path)
            logging.info("Created hardlink %s -> %s", target_path, source_path)
        except OSError:
            # Avoid accidentally duplicating multi-GB checkpoints when hardlinks are unavailable.
            try:
                size_bytes = source_path.stat().st_size
            except OSError:
                size_bytes = 0
            effective_max_copy_bytes = default_max_copy_bytes if max_copy_bytes is None else max_copy_bytes
            if effective_max_copy_bytes >= 0 and size_bytes > effective_max_copy_bytes:
                logging.warning(
                    "Could not hardlink %s -> %s and checkpoint is large (%.2f GB). "
                    "Skipping copy; rely on variant loading instead.",
                    target_path,
                    source_path,
                    size_bytes / (1024**3),
                )
                return
            shutil.copyfile(source_path, target_path)
            logging.info("Copied %s -> %s", source_path, target_path)

    ensure_weight_alias("text_encoder/model.fp16.safetensors", "text_encoder/model.safetensors", max_copy_bytes=-1)
    ensure_weight_alias("text_encoder_2/model.fp16.safetensors", "text_encoder_2/model.safetensors", max_copy_bytes=-1)
    # UNet checkpoints are often 5GB+; do not duplicate them automatically.
    ensure_weight_alias("unet/diffusion_pytorch_model.fp16.safetensors", "unet/diffusion_pytorch_model.safetensors", max_copy_bytes=0)
    ensure_weight_alias("vae/diffusion_pytorch_model.fp16.safetensors", "vae/diffusion_pytorch_model.safetensors", max_copy_bytes=-1)


def collect_style_images(base_dir: Path) -> List[Tuple[str, Path]]:
    images: List[Tuple[str, Path]] = []
    for style_dir in STYLE_NAMES:
        folder = base_dir / style_dir
        if not folder.exists():
            continue
        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file():
                continue
            suffix = image_path.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            rel = image_path.relative_to(base_dir)
            images.append((style_dir, rel))
    return images


def describe_image(image_path: Path) -> Dict[str, float]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im.resize((128, 128)), dtype=np.float32) / 255.0
    mean_rgb = arr.mean(axis=(0, 1))
    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = math.sqrt(float((rg ** 2).mean() + (yb ** 2).mean()))
    brightness = arr.mean()
    tone = "dark" if brightness < 0.4 else "bright" if brightness > 0.7 else "balanced"
    return {
        "colorfulness": float(colorfulness),
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "tone": tone,
    }


def call_qwen_caption(
    image_path: Path,
    style: str,
    qwen_url: str,
    guidance: str,
    caption_style: str,
) -> str:
    import requests

    stats = describe_image(image_path)
    prompt = (
        f"Style: {style}\n"
        f"Style guidance:\n{guidance}\n"
        f"Image tone: {stats['tone']}, colorfulness {stats['colorfulness']:.3f}, "
        f"mean RGB = ({stats['mean_r']:.2f}, {stats['mean_g']:.2f}, {stats['mean_b']:.2f}).\n"
        f"Task: Compose a concise caption (25-45 characters) that mentions the technique or color mood and "
        f"fits in a {caption_style} genre.\n"
        "Return the caption only, without numbering or additional commentary."
    )
    payload = {
        "model": "Qwen3-1.7B",
        "messages": [
            {
                "role": "system",
                "content": "You are a curator describing classical Chinese landscape paintings.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }
    for attempt in range(3):
        try:
            res = requests.post(qwen_url, json=payload, timeout=30)
            res.raise_for_status()
            resp_json = res.json()
            choices = resp_json.get("choices", [])
            if choices:
                caption = choices[0].get("message", {}).get("content", "").strip()
                if caption:
                    return caption.replace("\n", " ").strip()
        except Exception as exc:  # noqa: BLE002
            logging.warning("Qwen caption request failed (%s). attempt=%d", exc, attempt + 1)
            time.sleep(2)
    return f"{style} landscape"


def ensure_caption_table(
    dataset_dir: Path, table_path: Path, qwen_url: str, guidance: str, force: bool
) -> Dict[str, CaptionEntry]:
    logging.info("Loading caption table from %s", table_path)
    entries = {} if force or not table_path.exists() else read_caption_table(table_path)
    images = collect_style_images(dataset_dir)
    logging.info("Dataset scan found %d images under %s", len(images), dataset_dir)
    logging.info("Caption table currently has %d rows", len(entries))

    pending: List[Tuple[str, Path]] = []
    for style, rel in images:
        key = f"{style}|{rel.as_posix()}"
        if force or key not in entries:
            pending.append((style, rel))

    if pending:
        logging.info("Generating %d missing captions via Qwen (%s)", len(pending), qwen_url)
    else:
        logging.info("All images already have captions.")

    modified = False
    for index, (style, rel) in enumerate(pending, start=1):
        if index == 1 or index % 50 == 0:
            logging.info("Caption progress: %d/%d", index, len(pending))
        key = f"{style}|{rel.as_posix()}"
        caption = call_qwen_caption(dataset_dir / rel, style, qwen_url, guidance, style)
        entries[key] = CaptionEntry(style=style, relative_path=rel.as_posix(), caption=caption, last_updated=int(time.time()))
        modified = True
    if modified or force:
        write_caption_table(table_path, entries.values())
    return entries


class PaintingCaptionDataset(Dataset):
    def __init__(
        self,
        entries: List[CaptionEntry],
        root_dir: Path,
        resolution: int,
    ):
        self.entries = entries
        self.root_dir = root_dir
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[index]
        image_path = self.root_dir / entry.relative_path
        image = Image.open(image_path).convert("RGB")
        pixel_values = preprocess_image(image, self.resolution)
        return {
            "pixel_values": pixel_values,
            "caption": entry.caption,
        }


def preprocess_image(image: Image.Image, resolution: int) -> torch.Tensor:
    width, height = image.size
    scale = resolution / min(width, height)
    new_width = max(resolution, int(round(width * scale)))
    new_height = max(resolution, int(round(height * scale)))
    image = image.resize((new_width, new_height), resample=Image.BICUBIC)
    left = max(0, (new_width - resolution) // 2)
    top = max(0, (new_height - resolution) // 2)
    image = image.crop((left, top, left + resolution, top + resolution))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor

def train_loRA(args: argparse.Namespace, caption_entries: List[CaptionEntry]) -> None:
    """
    Train SDXL UNet LoRA adapters on the sorted dataset.

    Notes:
    - This training path targets the SDXL models you already have under `models/`.
    - It trains UNet attention LoRA only (no text-encoder LoRA) to keep dependencies and VRAM lower.
    """
    if not caption_entries:
        raise ValueError("Caption table is empty; cannot train.")

    # Prevent transformers from importing TensorFlow/JAX by default (avoids noisy oneDNN logs and speeds up imports).
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from accelerate import Accelerator  # type: ignore
        from diffusers import DDPMScheduler, StableDiffusionXLPipeline  # type: ignore
        from diffusers.models import attention_processor as attention_processor  # type: ignore
        from transformers import get_scheduler  # type: ignore
        from tqdm.auto import tqdm  # type: ignore
    except Exception as exc:  # noqa: BLE002
        raise RuntimeError(
            "Missing training dependencies. Install: "
            "pip install diffusers transformers accelerate tqdm peft"
        ) from exc

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")

    dtype = torch.float16 if args.fp16 and torch.cuda.is_available() else torch.float32
    logging.info("Loading SDXL pipeline from %s", args.pretrained_model)
    model_variant: Optional[str] = None
    pretrained_path = Path(args.pretrained_model)
    if pretrained_path.is_dir():
        ensure_sdxl_weight_aliases(pretrained_path)

        # Many offline SDXL dumps only contain fp16 weights, e.g.
        # - text_encoder/model.fp16.safetensors
        # - unet/diffusion_pytorch_model.fp16.safetensors
        # Without `variant="fp16"`, diffusers will look for non-variant filenames and fail.
        needs_fp16_variant = False
        for relative in (
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/model.fp16.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ):
            if (pretrained_path / relative).exists():
                needs_fp16_variant = True
                break
        if needs_fp16_variant:
            model_variant = "fp16"

    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=model_variant,
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model,
            torch_dtype=dtype,
            variant=model_variant,
        )
    pipe.to(accelerator.device)

    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    unet = pipe.unet

    # Attach LoRA to UNet (diffusers APIs differ across versions)
    lora_layers: Optional[torch.nn.Module] = None
    AttnProcsLayers = getattr(attention_processor, "AttnProcsLayers", None)
    LoRAAttnProcessor = getattr(attention_processor, "LoRAAttnProcessor", None)
    use_legacy_attn_procs = False
    if AttnProcsLayers is not None and LoRAAttnProcessor is not None:
        try:
            _ = LoRAAttnProcessor(hidden_size=1, cross_attention_dim=None, rank=1)  # type: ignore[call-arg]
            use_legacy_attn_procs = True
        except TypeError:
            use_legacy_attn_procs = False

    if use_legacy_attn_procs:
        # Older diffusers: LoRA attention processors are created per attention module.
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            is_self_attn = name.endswith("attn1.processor")
            cross_attention_dim = None if is_self_attn else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                continue
            lora_attn_procs[name] = LoRAAttnProcessor(  # type: ignore[misc,call-arg]
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=args.lora_rank,
            )
        unet.set_attn_processor(lora_attn_procs)
        lora_layers = AttnProcsLayers(unet.attn_processors)  # type: ignore[misc]
        params_to_optimize = lora_layers.parameters()
    else:
        # Newer diffusers: LoRA adapters are managed via the PEFT adapter API.
        try:
            from peft import LoraConfig  # type: ignore
        except Exception as exc:  # noqa: BLE002
            raise RuntimeError(
                "Your installed diffusers requires PEFT for LoRA training. Install: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        unet.add_adapter(lora_config)
        unet.enable_adapters()
        params_to_optimize = [p for p in unet.parameters() if p.requires_grad]

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    dataset = PaintingCaptionDataset(entries=caption_entries, root_dir=args.dataset, resolution=args.resolution)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = args.max_train_steps or math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.epochs
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    if lora_layers is not None:
        unet, lora_layers, optimizer, train_dataloader = accelerator.prepare(unet, lora_layers, optimizer, train_dataloader)
    else:
        unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)
    progress_bar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process, desc="Training SDXL LoRA")

    def save_checkpoint(step: int) -> None:
        if not accelerator.is_main_process:
            return
        target = args.lora_dir / f"{args.checkpoint_name}_step{step:04d}"
        target.mkdir(parents=True, exist_ok=True)
        logging.info("Saving UNet LoRA to %s", target)
        unwrapped_unet = accelerator.unwrap_model(unet)
        if hasattr(unwrapped_unet, "save_attn_procs"):
            unwrapped_unet.save_attn_procs(target / "unet_lora")
        elif lora_layers is not None and hasattr(accelerator.unwrap_model(lora_layers), "save_pretrained"):
            accelerator.unwrap_model(lora_layers).save_pretrained(target / "unet_lora")
        else:
            raise RuntimeError("Could not find a supported LoRA saving method for this diffusers version.")

    global_step = 0
    unet.train()

    # SDXL additional conditioning time_ids: [orig_w, orig_h, crop_x, crop_y, target_w, target_h]
    add_time_ids = torch.tensor(
        [args.resolution, args.resolution, 0, 0, args.resolution, args.resolution],
        device=accelerator.device,
        dtype=torch.long,
    )

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dataloader):
            if global_step >= total_steps:
                break
            captions: List[str] = batch["caption"]
            with accelerator.accumulate(unet):
                with accelerator.autocast():
                    pixel_values = batch["pixel_values"].to(accelerator.device, dtype=dtype)
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor

                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device
                    ).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                        prompt=captions,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                    time_ids_batch = add_time_ids.unsqueeze(0).repeat(latents.shape[0], 1)
                    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids_batch}

                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample

                    if noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=float(loss.detach().item()), step=global_step)

                if args.save_steps and global_step % args.save_steps == 0:
                    save_checkpoint(global_step)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    accelerator.wait_for_everyone()
    save_checkpoint(global_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption generation + LoRA fine-tuning for sorted style images.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"D:\大三上\AIGC\Chinese-Landscape-Painting-Dataset\sorted_by_style"),
        help="Folder containing the three style subdirectories (水墨, 工笔, 青绿).",
    )
    parser.add_argument(
        "--caption-table",
        type=Path,
        default=Path("LoRA/captions.csv"),
        help="CSV file used to store the generated captions.",
    )
    parser.add_argument(
        "--qwen-url",
        type=str,
        default="http://127.0.0.1:8001/v1/chat/completions",
        help="Endpoint of the local Qwen3 chat completion service.",
    )
    parser.add_argument(
        "--caption-force",
        action="store_true",
        help="Force regeneration of every caption (overwrites the CSV).",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=str(Path("models") / "stable-diffusion-xl-base-1.0"),
        help="Local SDXL model path (recommended: one folder under models/).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run.")
    parser.add_argument("--resolution", type=int, default=512, help="Square resolution to resize images to.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many updates (0 => derive from epochs).")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lr-scheduler", type=str, default="linear")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-workers", "--num-worker", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100, help="Periodic checkpoint interval.")
    parser.add_argument("--checkpoint-name", type=str, default="style_adapter", help="Prefix for saved adapters.")
    parser.add_argument("--lora-dir", type=Path, default=Path("LoRA"), help="Where to save LoRA adapters + tables.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training when CUDA is available.")
    parser.add_argument("--caption-only", action="store_true", help="Only generate/update captions and exit.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    sorted_dir = args.dataset
    if not sorted_dir.exists():
        raise FileNotFoundError(f"{sorted_dir} is missing; run scripts/style_sort.py first")
    entries = ensure_caption_table(
        dataset_dir=sorted_dir,
        table_path=args.caption_table,
        qwen_url=args.qwen_url,
        guidance=STYLE_GUIDANCE,
        force=args.caption_force,
    )
    if args.caption_only:
        logging.info("Caption table ready (%d entries).", len(entries))
        return
    args.lora_dir.mkdir(parents=True, exist_ok=True)
    train_loRA(args, list(entries.values()))


if __name__ == "__main__":
    main()
