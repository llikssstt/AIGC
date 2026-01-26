# LoRA Fine-tuning Workflow

Use `scripts/lora_finetune.py` to keep the captions for every sorted image synchronized with Qwen and to train LoRA adapters that specialize Stable Diffusion on your three style folders.

## Requirements

- The images live directly in `D:\大三上\AIGC\Chinese-Landscape-Painting-Dataset\sorted_by_style`, with `水墨`、`工笔`、`青绿` as immediate subdirectories. The script assumes this exact layout.
- Place your image-generation checkpoint under `models/` (recommended: `models/stable-diffusion-xl-base-1.0`). The script targets **SDXL** and expects the SDXL folder to contain `unet`, `text_encoder`, `text_encoder_2`, `tokenizer`, `tokenizer_2`, `vae`, and scheduler weights so `diffusers` can load it offline.
- If your local SDXL dump only contains fp16 weights (e.g. `text_encoder/model.fp16.safetensors`, `unet/diffusion_pytorch_model.fp16.safetensors`), the script will load with the `fp16` variant automatically; use `--fp16` during training for best speed/VRAM.
- If you see TensorFlow `oneDNN custom operations are on` messages, they are informational. The script disables TensorFlow/Flax imports in `transformers` by default to keep the logs quiet.
- Start the local Qwen3-1.7B server (`python -m sdxl_app.engine.simple_llm_server --model models/Qwen3-1.7B --port 8001`) so `http://127.0.0.1:8001/v1/chat/completions` is reachable.
- Install the Python dependencies: `pip install torch torchvision diffusers transformers accelerate requests numpy tqdm peft`.

## Step 1 – generate or refresh captions

The script writes `LoRA/captions.csv`, one row per image, by checking whether a caption already exists and asking Qwen only when needed.

```bash
python scripts/lora_finetune.py \
  --dataset "D:\大三上\AIGC\Chinese-Landscape-Painting-Dataset\sorted_by_style" \
  --caption-table LoRA/captions.csv \
  --qwen-url http://127.0.0.1:8001/v1/chat/completions
```

Use `--caption-force` to regenerate every row, or add `--caption-only` to stop once the table has been updated.

## Step 2 – LoRA training

When captions are ready, train LoRA adapters. Unlike the previous layout, this command points to the local `models/` checkpoint.

```bash
python scripts/lora_finetune.py \
  --dataset "D:\大三上\AIGC\Chinese-Landscape-Painting-Dataset\sorted_by_style" \
  --caption-table LoRA/captions.csv \
  --lora-dir LoRA \
  --pretrained-model "models/stable-diffusion-xl-base-1.0" \
  --batch-size 4 \
  --epochs 1 \
  --resolution 512 \
  --learning-rate 2e-4 \
  --save-steps 100
```

`--pretrained-model` defaults to `models/stable-diffusion-xl-base-1.0`, so you can omit it if you keep that folder name.

Key flags:

- `--fp16`: enable half precision (requires CUDA).
- `--max-train-steps`: stop after the given number of updates instead of relying on epochs.
- `--num-workers` / `--num-worker`: dataloader worker processes (use `0` on Windows if you hit worker crashes).
- `--save-steps`: checkpoints are saved under `LoRA/<checkpoint-name>_stepXXXX`.
- `--checkpoint-name`: friendly prefix that appears in each checkpoint folder.

## Outputs

- `LoRA/captions.csv`: the caption catalog that now matches the exact sorted_by_style dataset.
- `LoRA/<name>_stepXXXX/unet_lora/`: SDXL UNet LoRA weights (loadable via diffusers `load_attn_procs` / SDXL LoRA loading).
- Log messages mention whether the caption came from a heuristic or from the Qwen reply so you can audit borderline cases.

Re-run the script whenever you add new images, update captions, or experiment with different hyper-parameters or base models inside `models/`.
