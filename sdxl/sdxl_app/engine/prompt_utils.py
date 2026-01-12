from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBundle:
    global_prompt: str
    final_prompt: str
    negative_prompt: str


class PromptCard(BaseModel):
    edit_type: Literal["generate", "edit", "import"]

    style_preset: Optional[str] = None
    global_prompt: str = ""
    edit_text: Optional[str] = None

    final_prompt: str
    negative_prompt: str

    seed: int
    steps: int
    cfg: float

    width: Optional[int] = None
    height: Optional[int] = None
    strength: Optional[float] = None

    grow_pixels: Optional[int] = None
    blur_sigma: Optional[float] = None
    invert_mask: Optional[bool] = None

    extra: Dict[str, Any] = Field(default_factory=dict)


class PromptCompiler:
    def __init__(
        self,
        style_presets: Dict[str, str],
        negative_prompt: str,
        inpaint_negative_append: str,
    ):
        self.style_presets = dict(style_presets)
        self.negative_prompt = negative_prompt
        self.inpaint_negative_append = inpaint_negative_append

    def available_styles(self) -> list[str]:
        return sorted(self.style_presets.keys())

    def compile_generation(self, style_preset: str, scene_text: str) -> PromptBundle:
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        global_prompt = self.style_presets[style_preset].strip()
        scene_text = (scene_text or "").strip()

        final_prompt = f"{global_prompt}, {scene_text}" if scene_text else global_prompt
        neg = self.negative_prompt
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg)

    def compile_edit(self, global_prompt: str, edit_text: str) -> PromptBundle:
        global_prompt = (global_prompt or "").strip()
        edit_text = (edit_text or "").strip()

        if global_prompt and edit_text:
            final_prompt = f"{global_prompt}, {edit_text}, in the masked area only"
        elif global_prompt and not edit_text:
            final_prompt = f"{global_prompt}, in the masked area only"
        elif (not global_prompt) and edit_text:
            final_prompt = f"{edit_text}, in the masked area only"
        else:
            final_prompt = "in the masked area only"

        neg = self.negative_prompt + self.inpaint_negative_append
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg)

    def generation_card(
        self,
        *,
        style_preset: str,
        scene_text: str,
        bundle: PromptBundle,
        seed: int,
        steps: int,
        cfg: float,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        card = PromptCard(
            edit_type="generate",
            style_preset=style_preset,
            global_prompt=bundle.global_prompt,
            edit_text=scene_text,
            final_prompt=bundle.final_prompt,
            negative_prompt=bundle.negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            width=width,
            height=height,
        )
        return card.model_dump()

    def edit_card(
        self,
        *,
        style_preset: Optional[str],
        global_prompt: str,
        edit_text: str,
        bundle: PromptBundle,
        seed: int,
        steps: int,
        cfg: float,
        strength: float,
        grow_pixels: int,
        blur_sigma: float,
        invert_mask: bool,
    ) -> dict[str, Any]:
        card = PromptCard(
            edit_type="edit",
            style_preset=style_preset,
            global_prompt=global_prompt,
            edit_text=edit_text,
            final_prompt=bundle.final_prompt,
            negative_prompt=bundle.negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            strength=strength,
            grow_pixels=grow_pixels,
            blur_sigma=blur_sigma,
            invert_mask=invert_mask,
        )
        return card.model_dump()

    def import_card(self) -> dict[str, Any]:
        """
        导入图片版本的最小可复现卡片（无 prompt / 无采样参数）。
        """
        card = PromptCard(
            edit_type="import",
            style_preset=None,
            global_prompt="",
            edit_text=None,
            final_prompt="",
            negative_prompt="",
            seed=-1,
            steps=0,
            cfg=0.0,
        )
        return card.model_dump()
