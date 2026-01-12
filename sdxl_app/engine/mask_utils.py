from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessedMask:
    mask_for_inpaint: Image.Image  # L: 0/255，白=编辑区
    alpha: np.ndarray              # float32 (H,W) 0~1


def to_grayscale(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            alpha = mask[:, :, 3]
            if alpha.min() != alpha.max():
                mask = alpha
            else:
                mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return mask.astype(np.uint8)


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def grow_mask(mask: np.ndarray, pixels: int = 8) -> np.ndarray:
    if pixels <= 0:
        return mask
    k = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def feather_mask(mask: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    if sigma <= 0:
        return mask
    ksize = int(sigma * 6) | 1
    ksize = max(3, ksize)
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return 255 - mask


class MaskProcessor:
    def preprocess(self, mask: Image.Image, grow_pixels: int, blur_sigma: float, invert: bool) -> ProcessedMask:
        logger.info("Preprocessing mask: grow=%s blur=%s invert=%s", grow_pixels, blur_sigma, invert)

        rgba = mask.convert("RGBA")
        arr = np.array(rgba)
        gray = to_grayscale(arr)

        if invert:
            gray = invert_mask(gray)

        binary = binarize_mask(gray)
        grown = grow_mask(binary, grow_pixels)
        feathered = feather_mask(grown, blur_sigma)

        mask_for_inpaint = Image.fromarray(grown, mode="L")
        alpha = feathered.astype(np.float32) / 255.0
        return ProcessedMask(mask_for_inpaint=mask_for_inpaint, alpha=alpha)

    def alpha_blend(self, edited: Image.Image, original: Image.Image, alpha: np.ndarray) -> Image.Image:
        if edited.size != original.size:
            edited = edited.resize(original.size, Image.Resampling.LANCZOS)

        edited_arr = np.array(edited.convert("RGB"), dtype=np.float32)
        orig_arr = np.array(original.convert("RGB"), dtype=np.float32)

        if alpha.shape[:2] != edited_arr.shape[:2]:
            alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
            alpha_pil = alpha_pil.resize(edited.size, Image.Resampling.LANCZOS)
            alpha = np.array(alpha_pil, dtype=np.float32) / 255.0

        a = alpha[:, :, None]
        out = edited_arr * a + orig_arr * (1.0 - a)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")

