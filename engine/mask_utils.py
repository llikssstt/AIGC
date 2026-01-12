# ============================================================
# SDXL Inpainting Editor - Mask Processing Utilities
# ============================================================
"""
Mask 预处理模块：
- 灰度转换与二值化
- 形态学膨胀 (grow)
- 高斯羽化 (feather)
- Alpha blend 后处理
"""
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def to_grayscale(mask: np.ndarray) -> np.ndarray:
    """
    将 mask 转换为灰度图
    
    Args:
        mask: 输入 mask 数组（可以是 RGB、RGBA 或灰度）
        
    Returns:
        灰度 mask 数组 (H, W)，值范围 0-255
    """
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            # RGBA: 优先使用 alpha 通道（ImageEditor 常用透明度表示 mask）
            alpha = mask[:, :, 3]
            # 若 alpha 全部一致（例如整张不透明），回退到 RGB 转灰度
            if alpha.min() != alpha.max():
                mask = alpha
            else:
                mask = cv2.cvtColor(mask[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif mask.shape[2] == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return mask.astype(np.uint8)


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    二值化 mask：白色 (255) 表示编辑区，黑色 (0) 表示保留区
    
    Args:
        mask: 灰度 mask 数组
        threshold: 二值化阈值
        
    Returns:
        二值化后的 mask
    """
    _, binary = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return binary


def grow_mask(mask: np.ndarray, pixels: int = 8) -> np.ndarray:
    """
    形态学膨胀 mask（扩大编辑区域边界）
    
    Args:
        mask: 二值 mask
        pixels: 膨胀像素数
        
    Returns:
        膨胀后的 mask
    """
    if pixels <= 0:
        return mask
    
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    dilated = cv2.dilate(mask, kernel, iterations=1)
    logger.debug(f"Mask grown by {pixels} pixels")
    return dilated


def feather_mask(mask: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    """
    羽化 mask 边缘（高斯模糊实现软边缘过渡）
    
    Args:
        mask: 二值或灰度 mask
        sigma: 高斯模糊 sigma 值
        
    Returns:
        羽化后的 mask，值范围 0-255
    """
    if sigma <= 0:
        return mask
    
    # 计算 kernel size（必须为奇数）
    ksize = int(sigma * 6) | 1  # 确保奇数
    ksize = max(3, ksize)
    
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), sigma)
    feathered = np.clip(blurred, 0, 255).astype(np.uint8)
    logger.debug(f"Mask feathered with sigma={sigma}")
    return feathered


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """
    反转 mask（白变黑，黑变白）
    
    Args:
        mask: 输入 mask
        
    Returns:
        反转后的 mask
    """
    return 255 - mask


def preprocess_mask(
    mask: np.ndarray | Image.Image,
    grow_pixels: int = 8,
    blur_sigma: float = 12.0,
    invert: bool = False
) -> Tuple[Image.Image, np.ndarray]:
    """
    完整的 mask 预处理流程
    
    Args:
        mask: 输入 mask（可以是 PIL Image 转的数组，或直接 numpy 数组）
        grow_pixels: 膨胀像素数
        blur_sigma: 羽化 sigma
        invert: 是否反转 mask
        
    Returns:
        Tuple of:
            - PIL Image: 用于 SDXL inpaint 的 mask（二值，白=编辑区）
            - np.ndarray: 用于后处理 alpha blend 的羽化 mask（0-1 浮点）
    """
    logger.info(f"Preprocessing mask: grow={grow_pixels}, blur={blur_sigma}, invert={invert}")

    # 允许直接传入 PIL.Image（例如 FastAPI UploadFile 读取出来的 PNG）
    if isinstance(mask, Image.Image):
        # 统一转换为 numpy，保留 alpha（若存在）
        mask = np.array(mask.convert("RGBA"))
    
    # Step 1: 转灰度
    gray = to_grayscale(mask)
    
    # Step 2: 可选反转
    if invert:
        gray = invert_mask(gray)
        logger.debug("Mask inverted")
    
    # Step 3: 二值化
    binary = binarize_mask(gray)
    
    # Step 4: 膨胀
    grown = grow_mask(binary, grow_pixels)
    
    # Step 5: 羽化（用于后处理）
    feathered = feather_mask(grown, blur_sigma)
    
    # 转换为 PIL Image（SDXL 需要）
    mask_pil = Image.fromarray(grown, mode='L')
    
    # 归一化羽化 mask 到 0-1（用于 alpha blend）
    alpha_mask = feathered.astype(np.float32) / 255.0
    
    logger.info(f"Mask preprocessing complete: shape={mask_pil.size}")
    return mask_pil, alpha_mask


def alpha_blend(
    edited: Image.Image,
    original: Image.Image,
    alpha_mask: np.ndarray
) -> Image.Image:
    """
    Alpha 混合：用羽化 mask 融合编辑图和原图
    
    公式: out = edited * alpha + original * (1 - alpha)
    
    目的：最大化保持非 mask 区域不变，边缘自然过渡
    
    Args:
        edited: 编辑后的图像
        original: 原始图像
        alpha_mask: 羽化后的 alpha mask（0-1 浮点数组）
        
    Returns:
        融合后的图像
    """
    logger.info("Performing alpha blend post-processing")
    
    # 确保尺寸一致
    if edited.size != original.size:
        edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    
    # 转 numpy
    edited_arr = np.array(edited).astype(np.float32)
    original_arr = np.array(original).astype(np.float32)
    
    # 调整 alpha_mask 形状
    if alpha_mask.shape[:2] != edited_arr.shape[:2]:
        # 需要 resize
        alpha_pil = Image.fromarray((alpha_mask * 255).astype(np.uint8), mode='L')
        alpha_pil = alpha_pil.resize(edited.size, Image.Resampling.LANCZOS)
        alpha_mask = np.array(alpha_pil).astype(np.float32) / 255.0
    
    # 扩展 alpha 到 3 通道
    if edited_arr.ndim == 3 and alpha_mask.ndim == 2:
        alpha_3ch = alpha_mask[:, :, np.newaxis]
    else:
        alpha_3ch = alpha_mask
    
    # Alpha blend
    blended = edited_arr * alpha_3ch + original_arr * (1 - alpha_3ch)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    result = Image.fromarray(blended)
    logger.info("Alpha blend complete")
    return result


def create_mask_from_points(
    image_size: Tuple[int, int],
    points: list,
    brush_size: int = 20
) -> np.ndarray:
    """
    从画笔点列表创建 mask（用于简单的程序化 mask 生成）
    
    Args:
        image_size: (width, height)
        points: 点坐标列表 [(x1, y1), (x2, y2), ...]
        brush_size: 画笔大小
        
    Returns:
        mask 数组
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(len(points) - 1):
        pt1 = (int(points[i][0]), int(points[i][1]))
        pt2 = (int(points[i + 1][0]), int(points[i + 1][1]))
        cv2.line(mask, pt1, pt2, 255, thickness=brush_size)
    
    return mask


def validate_mask(mask: np.ndarray, min_pixels: int = 100) -> bool:
    """
    验证 mask 是否有效（有足够的编辑区域）
    
    Args:
        mask: mask 数组
        min_pixels: 最小白色像素数
        
    Returns:
        是否有效
    """
    gray = to_grayscale(mask)
    white_pixels = np.sum(gray > 127)
    is_valid = white_pixels >= min_pixels
    
    if not is_valid:
        logger.warning(f"Mask validation failed: only {white_pixels} white pixels (min: {min_pixels})")
    
    return is_valid
