# ============================================================
# SDXL Inpainting Editor - Prompt Utilities
# ============================================================
"""
Prompt 构建与管理模块：
- 风格预设管理
- 生成 prompt 构建
- 编辑 prompt 构建
- Prompt 卡片生成
"""
import logging
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


# ============================================================
# 风格预设（国风三套）
# ============================================================
STYLE_PRESETS: Dict[str, str] = {
    "水墨": (
        "traditional Chinese ink wash painting, shuimo style, "
        "flowing ink strokes, elegant brushwork, monochrome with subtle color gradients, "
        "misty atmosphere, xieyi freehand style, rice paper texture, "
        "masterpiece, best quality, highly detailed"
    ),
    "工笔": (
        "Chinese gongbi meticulous painting style, fine brushwork, "
        "delicate lines, rich colors, detailed rendering, "
        "silk painting texture, court painting style, "
        "exquisite details, traditional pigments, "
        "masterpiece, best quality, highly detailed"
    ),
    "青绿": (
        "Chinese qinglv landscape painting, blue-green landscape style, "
        "mineral pigments, azurite blue and malachite green, "
        "Tang dynasty style, golden outlines, layered mountains, "
        "decorative clouds, panoramic composition, "
        "masterpiece, best quality, highly detailed"
    ),
}


# ============================================================
# 默认负面提示词
# ============================================================
DEFAULT_NEGATIVE_PROMPT: str = (
    "watermark, text, logo, signature, username, "
    "low quality, worst quality, blurry, pixelated, "
    "deformed, ugly, bad anatomy, extra limbs, "
    "photorealistic, 3d render, photography, "
    "modern elements, western style, "
    "nsfw, nude"
)

# Inpainting 专用负面提示词追加
INPAINT_NEGATIVE_APPEND: str = (
    ", change background, outside mask changes, "
    "extra objects outside region, affect unmasked area, "
    "inconsistent style, different lighting"
)


# ============================================================
# Prompt 卡片数据结构
# ============================================================
@dataclass
class PromptCard:
    """Prompt 卡片：记录完整的生成/编辑参数，便于复现"""
    global_prompt: str
    edit_text: Optional[str]
    final_prompt: str
    negative_prompt: str
    seed: int
    steps: int
    cfg: float
    strength: Optional[float]  # 仅 inpaint 有
    height: Optional[int]
    width: Optional[int]
    grow_pixels: Optional[int]
    blur_sigma: Optional[float]
    style_preset: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
    
    def to_display_text(self) -> str:
        """生成可读的显示文本"""
        lines = [
            "═" * 50,
            "📋 Prompt Card",
            "═" * 50,
            f"🎨 Style: {self.style_preset or 'Custom'}",
            f"",
            f"📝 Global Prompt:",
            f"   {self.global_prompt[:100]}..." if len(self.global_prompt) > 100 else f"   {self.global_prompt}",
        ]
        
        if self.edit_text:
            lines.extend([
                f"",
                f"✏️ Edit Text:",
                f"   {self.edit_text}",
            ])
        
        lines.extend([
            f"",
            f"🔧 Final Prompt:",
            f"   {self.final_prompt[:150]}..." if len(self.final_prompt) > 150 else f"   {self.final_prompt}",
            f"",
            f"🚫 Negative Prompt:",
            f"   {self.negative_prompt[:100]}..." if len(self.negative_prompt) > 100 else f"   {self.negative_prompt}",
            f"",
            "─" * 50,
            f"⚙️ Parameters:",
            f"   • Seed: {self.seed}",
            f"   • Steps: {self.steps}",
            f"   • CFG Scale: {self.cfg}",
        ])
        
        if self.strength is not None:
            lines.append(f"   • Strength (Denoise): {self.strength}")
        
        if self.height and self.width:
            lines.append(f"   • Size: {self.width} × {self.height}")
        
        if self.grow_pixels is not None:
            lines.append(f"   • Mask Grow: {self.grow_pixels}px")
        
        if self.blur_sigma is not None:
            lines.append(f"   • Mask Blur: {self.blur_sigma}")
        
        lines.append("═" * 50)
        
        return "\n".join(lines)


# ============================================================
# Prompt 构建函数
# ============================================================
def get_style_prompt(style: str) -> str:
    """
    获取风格预设 prompt
    
    Args:
        style: 风格名称（水墨/工笔/青绿）
        
    Returns:
        风格 prompt 字符串
        
    Raises:
        ValueError: 未知风格
    """
    if style not in STYLE_PRESETS:
        available = ", ".join(STYLE_PRESETS.keys())
        raise ValueError(f"Unknown style '{style}'. Available: {available}")
    
    return STYLE_PRESETS[style]


def build_generation_prompt(
    style: str,
    scene_text: str
) -> Tuple[str, str, str]:
    """
    构建文生图 prompt
    
    Args:
        style: 风格预设名称
        scene_text: 用户输入的场景描述
        
    Returns:
        Tuple of (global_prompt, final_prompt, negative_prompt)
    """
    global_prompt = get_style_prompt(style)
    
    # 合并场景描述
    if scene_text.strip():
        final_prompt = f"{global_prompt}, {scene_text.strip()}"
    else:
        final_prompt = global_prompt
    
    negative_prompt = DEFAULT_NEGATIVE_PROMPT
    
    logger.info(f"Built generation prompt for style '{style}'")
    logger.debug(f"Final prompt: {final_prompt[:100]}...")
    
    return global_prompt, final_prompt, negative_prompt


def build_edit_prompt(
    global_prompt: str,
    edit_text: str
) -> Tuple[str, str]:
    """
    构建 Inpainting 编辑 prompt
    
    公式: final_prompt = global_prompt + ", " + edit_text + ", in the masked area only"
    
    Args:
        global_prompt: 全局风格 prompt（从 session 继承）
        edit_text: 用户输入的编辑指令
        
    Returns:
        Tuple of (final_prompt, negative_prompt)
    """
    # 构建 final prompt
    edit_text = edit_text.strip()
    if edit_text:
        final_prompt = f"{global_prompt}, {edit_text}, in the masked area only"
    else:
        final_prompt = f"{global_prompt}, in the masked area only"
    
    # Inpainting 专用负面提示词
    negative_prompt = DEFAULT_NEGATIVE_PROMPT + INPAINT_NEGATIVE_APPEND
    
    logger.info(f"Built edit prompt with edit_text: '{edit_text}'")
    logger.debug(f"Final prompt: {final_prompt[:100]}...")
    
    return final_prompt, negative_prompt


def create_generation_card(
    style: str,
    scene_text: str,
    global_prompt: str,
    final_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    height: int,
    width: int
) -> PromptCard:
    """
    创建文生图 Prompt 卡片
    """
    return PromptCard(
        global_prompt=global_prompt,
        edit_text=scene_text,
        final_prompt=final_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        strength=None,
        height=height,
        width=width,
        grow_pixels=None,
        blur_sigma=None,
        style_preset=style
    )


def create_edit_card(
    global_prompt: str,
    edit_text: str,
    final_prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    cfg: float,
    strength: float,
    grow_pixels: int,
    blur_sigma: float,
    style_preset: Optional[str] = None
) -> PromptCard:
    """
    创建编辑 Prompt 卡片
    """
    return PromptCard(
        global_prompt=global_prompt,
        edit_text=edit_text,
        final_prompt=final_prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        strength=strength,
        height=None,
        width=None,
        grow_pixels=grow_pixels,
        blur_sigma=blur_sigma,
        style_preset=style_preset
    )


def get_available_styles() -> list:
    """获取所有可用风格列表"""
    return list(STYLE_PRESETS.keys())
