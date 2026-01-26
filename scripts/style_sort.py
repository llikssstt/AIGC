"""
Sort the Chinese Landscape Painting dataset into 水墨，工笔，青绿 style folders by
asking a local Qwen3-1.7B model to interpret lightweight image statistics.

The script extracts a few color/contrast features from each painting, summarizes them,
and sends the summary to the Qwen model. The returned category is mapped to one of the
three folders and the image is copied with a normalized filename.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

STYLES: Tuple[str, ...] = ("水墨", "工笔", "青绿")
STYLE_PREFIX: Dict[str, str] = {"水墨": "shimo", "工笔": "gongbi", "青绿": "qinglv"}
STYLE_GUIDANCE = (
    "水墨：以墨为主，发展墨韵、留白与虚实，色彩稀少、灰度比例高，笔触强调气的流动。\n"
    "工笔：以精细描绘为核心，线条清晰、结构准确，细节清楚并可能伴有层层设色与罩染。\n"
    "青绿：以矿物颜料（石青、石绿）为主，色彩鲜明，可能对应工笔青绿或写意青绿。"
)


class QwenStyleSorter:
    """Wraps the Qwen3 inference logic used to assign a style label."""

    def __init__(
        self,
        model_path: Path,
        device: str,
        max_tokens: int = 80,
        temperature: float = 0.2,
    ):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        logging.info("Loading Qwen3 model from %s on %s", self.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def summarize_image(self, image_path: Path) -> Dict[str, float]:
        with Image.open(image_path) as raw:
            rgb = raw.convert("RGB")
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        height, width = arr.shape[:2]
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).mean()
        max_channel = arr.max(axis=2)
        min_channel = arr.min(axis=2)
        saturation = np.where(max_channel == 0, 0, (max_channel - min_channel) / max_channel).mean()
        gray_ratio = (np.std(arr, axis=2) < 0.03).mean()
        rg = arr[..., 0] - arr[..., 1]
        yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
        colorfulness = math.sqrt(float((rg ** 2).mean() + (yb ** 2).mean()))
        blue_green_ratio = (mean_rgb[1] + mean_rgb[2]) / (mean_rgb[0] + 1e-6)

        return {
            "width": width,
            "height": height,
            "mean_r": float(mean_rgb[0]),
            "mean_g": float(mean_rgb[1]),
            "mean_b": float(mean_rgb[2]),
            "std_rgb": float(std_rgb.mean()),
            "luminance": float(lum),
            "saturation": float(saturation),
            "gray_ratio": float(gray_ratio),
            "colorfulness": float(colorfulness),
            "blue_green_ratio": float(blue_green_ratio),
        }

    def format_summary(self, features: Dict[str, float]) -> str:
        lines: List[str] = [
            f"Dimensions: {features['width']}x{features['height']}",
            f"Average RGB (0-1): R={features['mean_r']:.2f}, G={features['mean_g']:.2f}, B={features['mean_b']:.2f}",
            f"Mean deviation (per channel): {features['std_rgb']:.3f}",
            f"Luminance (0-1): {features['luminance']:.3f}",
            f"Saturation (0-1): {features['saturation']:.3f}",
            f"Gray-pixel ratio (<0.03 std): {features['gray_ratio']:.2%}",
            f"Colorfulness score: {features['colorfulness']:.3f}",
            f"Blue/green emphasis (G+B)/R: {features['blue_green_ratio']:.3f}",
        ]
        return "\n".join(lines)

    def _heuristic_style(self, features: Dict[str, float]) -> Tuple[Optional[str], str]:
        if features["colorfulness"] < 0.08 and features["gray_ratio"] > 0.62:
            return "水墨", "low colorfulness with high gray ratio indicates ink focus"
        if features["blue_green_ratio"] > 1.2 and features["colorfulness"] > 0.18:
            return "青绿", "strong blue/green emphasis and visible mineral color"
        if features["saturation"] > 0.2 and features["colorfulness"] > 0.12:
            return "工笔", "measured colors with clear saturation suggesting fine rendering"
        return None, ""

    def classify(self, image_path: Path) -> Tuple[str, str]:
        features = self.summarize_image(image_path)
        summary = self.format_summary(features)
        heuristic_style, heuristic_reason = self._heuristic_style(features)

        if heuristic_style:
            logging.info(
                "Heuristic classification for %s => %s (%s)", image_path.name, heuristic_style, heuristic_reason
            )
            return heuristic_style, f"Heuristic: {heuristic_reason}"
        system = (
            "You are an art historian specialising in traditional Chinese landscape painting. "
            "Rely only on the textual features provided."
        )
        user = (
            f"Features:\n{summary}\n\n"
            f"Guidance:\n{STYLE_GUIDANCE}\n\n"
            "From the list [水墨, 工笔, 青绿], pick the category that best fits the work. "
            "Answer in two lines: a one-sentence rationale followed by 'Style: <name>'. "
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        device = torch.device(self.device)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            generated = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated[0][input_len:]
        result = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        logging.debug("LLM response: %s", result)

        chosen = self._parse_style(result)
        logging.info("Mapped %s => %s (%s)", image_path.name, chosen, result.replace("\n", " | "))
        return chosen, result

    @staticmethod
    def _parse_style(generated: str) -> str:
        for style in STYLES:
            if style in generated:
                return style
        # fallback heuristic
        if "墨" in generated or "黑" in generated:
            return "水墨"
        return STYLES[1]


def prepare_output_dirs(base_output: Path) -> Dict[str, Path]:
    result = {}
    for style in STYLES:
        dir_path = base_output / style
        dir_path.mkdir(parents=True, exist_ok=True)
        result[style] = dir_path
    return result


def collect_images(data_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort the Chinese Landscape Painting dataset by style using Qwen3."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("Chinese-Landscape-Painting-Dataset"),
        help="Root folder that contains the raw paintings.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(r"E:\AIGC\models\Qwen3-1.7B"),
        help="Local path to the Qwen3-1.7B model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Chinese-Landscape-Painting-Dataset/sorted_by_style"),
        help="Destination folder for the sorted dataset.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Run the Qwen3 model on CPU or CUDA.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum tokens to generate for each classification prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for Qwen3 (0 => greedy).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned copies without performing them.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    sorter = QwenStyleSorter(
        model_path=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    sorter.load_model()

    images = collect_images(args.dataset)
    if not images:
        logging.warning("No supported images were found under %s", args.dataset)
        return

    output_dirs = prepare_output_dirs(args.output)
    counters: Dict[str, int] = defaultdict(int)

    for image_path in images:
        style, reasoning = sorter.classify(image_path)
        counters[style] += 1
        suffix = image_path.suffix.lower()
        dest = output_dirs[style] / f"{STYLE_PREFIX[style]}_{counters[style]:05d}{suffix}"

        if args.dry_run:
            logging.info("Dry run: would copy %s -> %s (%s)", image_path, dest, style)
            continue

        shutil.copy2(image_path, dest)
        logging.debug("Copied %s to %s", image_path, dest)

    logging.info("Sorting complete: %s", ", ".join(f"{style}={count}" for style, count in counters.items()))


if __name__ == "__main__":
    main()
