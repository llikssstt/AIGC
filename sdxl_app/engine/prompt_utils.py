# ============================================================
# Prompt Utils - 纯 LLM 版本（无静态词典）
# ============================================================
"""
使用 LLM 理解中文古诗并生成 SDXL prompt。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal, TYPE_CHECKING
import re

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .llm_service import LLMService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptBundle:
    global_prompt: str
    final_prompt: str
    negative_prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)


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
    """纯 LLM 驱动的 Prompt 编译器"""

    def __init__(
        self,
        style_presets: Dict[str, str],
        negative_prompt: str,
        inpaint_negative_append: str,
        *,
        poetry_enabled: bool = True,
        poetry_preamble: str = (
            "poetic scene, visual storytelling, interpret the imagery and mood, "
            "do not include any text or calligraphy"
        ),
        poetry_negative_append: str = ", calligraphy, chinese characters, letters, words, subtitles",
        llm_service: Optional["LLMService"] = None,
    ):
        self.style_presets = dict(style_presets)
        self.negative_prompt = negative_prompt
        self.inpaint_negative_append = inpaint_negative_append
        self.poetry_enabled = poetry_enabled
        self.poetry_preamble = poetry_preamble
        self.poetry_negative_append = poetry_negative_append
        self.llm_service = llm_service

    def available_styles(self) -> list[str]:
        return sorted(self.style_presets.keys())

    def compile_generation(self, style_preset: str, scene_text: str) -> PromptBundle:
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        global_prompt = self.style_presets[style_preset].strip()
        scene_text = (scene_text or "").strip()

        # 检测是否为古诗/中文输入
        if self.poetry_enabled and self._looks_like_poetry(scene_text):
            return self.compile_poetry(style_preset=style_preset, poem_text=scene_text)

        # 普通英文输入
        final_prompt = f"{global_prompt}, {scene_text}" if scene_text else global_prompt
        neg = self.negative_prompt
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta={})

    def compile_poetry(self, *, style_preset: str, poem_text: str) -> PromptBundle:
        """
        使用 LLM 理解古诗并生成 prompt。
        如果 LLM 不可用，返回简化的基础 prompt。
        """
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        global_prompt = self.style_presets[style_preset].strip()
        poem_raw = (poem_text or "").strip()

        llm_error: Optional[str] = None
        llm_raw: str = ""

        # 使用 LLM 理解古诗
        if self.llm_service:
            try:
                llm_result = self.llm_service.interpret_poetry(poem_raw, style_preset)
                if llm_result:
                    llm_raw = (getattr(llm_result, "raw_response", "") or "")[:2000]
                    scene = (llm_result.scene_description or "").strip()
                    # 解析失败/输出不规范时，尽量从 raw_response 兜底一个可用的 scene
                    if not scene:
                        scene = self._salvage_scene_from_raw(getattr(llm_result, "raw_response", "")) or ""

                    parts = [global_prompt, self.poetry_preamble]
                    if scene:
                        parts.append(scene)
                    if llm_result.visual_elements:
                        parts.extend([e for e in llm_result.visual_elements[:5] if e])
                    if llm_result.mood:
                        parts.append(llm_result.mood)
                    parts.append("Chinese poetry scene")

                    # 只要 LLM 至少给出了 scene/elements/mood 任一项，就认为可用
                    if scene or llm_result.visual_elements or llm_result.mood:
                        final_prompt = ", ".join([p for p in parts if p])
                        neg = self.negative_prompt + self.poetry_negative_append
                        meta = {
                            "input_kind": "poetry_llm",
                            "poem_raw": poem_raw,
                            "llm_scene": scene,
                            "llm_elements": llm_result.visual_elements,
                            "llm_mood": llm_result.mood,
                            "llm_raw": llm_raw,
                        }
                        logger.info("LLM 古诗理解成功")
                        return PromptBundle(
                            global_prompt=global_prompt,
                            final_prompt=final_prompt,
                            negative_prompt=neg,
                            meta=meta,
                        )
            except Exception as e:
                llm_error = str(e)
                logger.warning("LLM 古诗理解失败: %s", e)

        # LLM 不可用/失败时：本地抽取一些常见意象，避免 prompt 过于空泛
        elements = self._extract_poetry_elements(poem_raw)
        parts = [global_prompt, self.poetry_preamble]
        parts.extend(elements[:6])
        parts.append("Chinese poetry scene")
        final_prompt = ", ".join([p for p in parts if p])
        neg = self.negative_prompt + self.poetry_negative_append
        meta: Dict[str, Any] = {
            "input_kind": "poetry_fallback",
            "poem_raw": poem_raw,
            "fallback_elements": elements[:12],
        }
        if llm_error:
            meta["llm_error"] = llm_error
        if llm_raw:
            meta["llm_raw"] = llm_raw
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta=meta)

    def _looks_like_poetry(self, text: str) -> bool:
        if not text:
            return False
        # 多行/常见诗词标点/明显中文字符 → 进入诗词模式
        if "\n" in text:
            return True
        if any(p in text for p in ("，", "。", "；", "！", "？")):
            return any("\u4e00" <= c <= "\u9fff" for c in text)
        # 纯中文短句
        return any("\u4e00" <= c <= "\u9fff" for c in text) and len(text) <= 80

    def _extract_poetry_elements(self, poem_raw: str) -> list[str]:
        """
        在没有 LLM 时，从古诗文本中做非常轻量的意象抽取。
        目标：给 SDXL 足够的视觉锚点（如 moon/frost/wine cup/shadow），避免仅剩泛化词。
        """
        text = (poem_raw or "").strip()
        if not text:
            return []

        rules: list[tuple[list[str], str]] = [
            (["明月", "望明月", "月"], "full moon"),
            (["月光"], "moonlight"),
            (["霜"], "frost"),
            (["举杯", "杯", "酒"], "wine cup"),
            (["对影", "影"], "shadow"),
            (["床前", "窗前"], "moonlight on floor"),
            (["故乡", "乡"], "homesickness"),
            (["夜"], "night"),
            (["雾", "烟"], "mist"),
            (["山"], "misty mountains"),
            (["江", "河", "水"], "river"),
        ]

        out: list[str] = []
        for keys, tag in rules:
            if any(k in text for k in keys) and tag not in out:
                out.append(tag)

        # 常见诗词默认补一个人物/氛围锚点（不强制）
        if any(k in text for k in ("我", "独", "思", "愁", "邀")) and "solitary scholar" not in out:
            out.append("solitary scholar")

        return out

    def _salvage_scene_from_raw(self, raw_response: str) -> str:
        """
        当模型未严格按 SCENE/ELEMENTS 格式输出时，尽量从原始文本中提取一个可用的 scene。
        """
        raw = (raw_response or "").strip()
        if not raw:
            return ""

        # 去掉可能的 code fence
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\\n|\\n```$", "", raw).strip()
        if not raw:
            return ""

        # 优先拿第一行/第一句作为 scene
        first_line = raw.splitlines()[0].strip()
        if not first_line:
            return ""

        # 如果第一行还是 "SCENE:" 这类前缀，剥掉
        first_line = re.sub(r"^(scene)\\s*[:：\\-]\\s*", "", first_line, flags=re.IGNORECASE).strip()
        # 控制长度：最多 25 个单词
        words = first_line.split()
        if len(words) > 25:
            first_line = " ".join(words[:25])
        return first_line

    def compile_edit(self, global_prompt: str, edit_text: str) -> PromptBundle:
        """使用 LLM 翻译编辑指令"""
        global_prompt = (global_prompt or "").strip()
        edit_text = (edit_text or "").strip()

        # 使用 LLM 翻译
        translated_edit = None
        if self.llm_service and edit_text:
            try:
                translated_edit = self.llm_service.translate_edit(edit_text)
                if translated_edit:
                    logger.info("LLM 翻译: %s -> %s", edit_text, translated_edit)
            except Exception as e:
                logger.warning("LLM 翻译失败: %s", e)

        # LLM 失败时，直接使用原文
        if not translated_edit:
            translated_edit = edit_text

        # 构建 prompt
        if translated_edit and global_prompt:
            final_prompt = f"{translated_edit}, {global_prompt}, in the masked area only"
        elif translated_edit:
            final_prompt = f"{translated_edit}, in the masked area only"
        elif global_prompt:
            final_prompt = f"{global_prompt}, in the masked area only"
        else:
            final_prompt = "in the masked area only"

        neg = self.negative_prompt + self.inpaint_negative_append
        meta = {"original_edit_text": edit_text, "translated_edit": translated_edit}
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta=meta)

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
        extra = dict(bundle.meta or {})
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
            extra=extra,
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
        extra = dict(bundle.meta or {})
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
            extra=extra,
        )
        return card.model_dump()

    def import_card(self) -> dict[str, Any]:
        """导入图片版本的最小卡片"""
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
