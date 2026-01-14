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

    """纯 LLM 驱动的 Prompt 编译器"""

    def __init__(
        self,
        style_presets: Dict[str, str],
        negative_prompt: str,
        inpaint_negative_append: str,
        *,
        poetry_enabled: bool = True,
        poetry_negative_append: str = ", calligraphy, chinese characters, letters, words, subtitles",
        llm_service: Optional["LLMService"] = None,
    ):
        self.style_presets = dict(style_presets)
        self.negative_prompt = negative_prompt
        self.inpaint_negative_append = inpaint_negative_append
        self.poetry_enabled = poetry_enabled
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
        使用 LLM 理解古诗并生成分层 Prompt (Layered Prompt)。
        """
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        # 基础风格 (Base Style) - 放在最后或作为底层
        global_style = self.style_presets[style_preset].strip()
        poem_raw = (poem_text or "").strip()

        llm_error: Optional[str] = None
        llm_raw: str = ""
        
        # 默认分层结构
        layers = {
            "subject": "",
            "action": "",
            "composition": "medium shot, centralized composition",
            "environment": "",
            "mood": "",
            "style": global_style,
        }
        elements = []
        interpretation_subject_description = "" # To store LLM's subject description for negative prompt logic
        llm_has_content = False

        if self.llm_service:
            try:
                interpretation = self.llm_service.interpret_poetry(poem_raw, style_preset)
                if interpretation:
                    llm_raw = interpretation.raw_response
                    interpretation_subject_description = interpretation.subject_description or ""
                    if interpretation_subject_description and interpretation_subject_description.lower() != "none":
                        llm_has_content = True
                    
                    # 1. Subject Layer (最为重要，加权)
                    if interpretation_subject_description and interpretation_subject_description.lower() not in ("none", "no humans"):
                        layers["subject"] = f"({interpretation_subject_description}:1.3)"
                    else:
                        layers["subject"] = "" # 明确无人

                    # 2. Action Layer
                    if interpretation.action_description and interpretation.action_description.lower() != "none":
                        layers["action"] = interpretation.action_description
                        llm_has_content = True

                    # 3. Environment Layer
                    if interpretation.environment_description:
                        layers["environment"] = interpretation.environment_description
                        llm_has_content = True

                    # 4. Composition Layer
                    if interpretation.composition_description:
                        layers["composition"] = interpretation.composition_description

                    # 5. Mood Layer
                    if interpretation.mood_description:
                        layers["mood"] = interpretation.mood_description
                        llm_has_content = True

                    # 6. Elements
                    elements = interpretation.visual_elements
                    if elements:
                        llm_has_content = True

                    logger.info("LLM 古诗分层理解成功: %s", layers)
            except Exception as e:
                llm_error = str(e)
                logger.warning("LLM 古诗理解失败: %s", e)

        # 如果 LLM 失败，回退到简单的规则抽取
        if not layers["subject"] and not layers["environment"] and not llm_has_content:
            fallback_elements = self._extract_poetry_elements(poem_raw)
            # 尝试简单归类
            env_keywords = ["mountains", "river", "moon", "snow", "mist"]
            subj_keywords = ["scholar", "fisherman", "boat", "old man"] # Added "old man"
            
            env_parts = [e for e in fallback_elements if any(k in e for k in env_keywords)]
            subj_parts = [e for e in fallback_elements if any(k in e for k in subj_keywords)]
            
            if subj_parts:
                layers["subject"] = f"({', '.join(subj_parts)}:1.2)"
            if env_parts:
                layers["environment"] = ", ".join(env_parts)
            
            elements = fallback_elements
            if llm_raw and not llm_error:
                llm_error = "LLM returned no usable fields; fallback applied."

        # === 组装 Prompt ===
        # 新顺序: Style (最重要，放首位) > Subject > Action > Environment > Mood > Composition
        prompt_parts = []
        
        # 1. 风格放在最前面，确保不会被截断
        prompt_parts.append(layers["style"])
        
        # 2. 主体和动作（LLM 核心输出）
        if layers["subject"]:
            prompt_parts.append(layers["subject"])
        
        if layers["action"]:
            prompt_parts.append(layers["action"])
            
        # 3. 环境和氛围
        if layers["environment"]:
            prompt_parts.append(layers["environment"])
            
        if layers["mood"]:
            prompt_parts.append(layers["mood"])

        # 4. 构图放在后面（可被截断）
        if layers["composition"]:
            prompt_parts.append(layers["composition"])
        
        # 5. 补充 Elements (去重，限制数量)
        used_text = " ".join(prompt_parts).lower()
        added_elements = 0
        for elem in elements:
            if elem.lower() not in used_text and added_elements < 2:
                prompt_parts.append(elem)
                added_elements += 1

        final_prompt = ", ".join([p for p in prompt_parts if p])
        
        # Token 限制：CLIP 最大 77 tokens (~320 chars)，从末尾截断
        MAX_PROMPT_CHARS = 320
        if len(final_prompt) > MAX_PROMPT_CHARS:
            final_prompt = final_prompt[:MAX_PROMPT_CHARS].rsplit(",", 1)[0]
            logger.warning("Prompt truncated to %d chars to fit CLIP limit", len(final_prompt))
        
        # 构建负面提示
        neg = self.negative_prompt + self.poetry_negative_append
        # 如果主体这一层明确是空的，确保负面提示里加上 no humans 强化无人
        if not layers["subject"] and "no humans" in interpretation_subject_description.lower():
             neg += ", humans, person, people, man, woman"
        elif layers["subject"]:
             # 有主体时，确保不要 prohibited "no humans"
             neg = neg.replace("no humans,", "").replace("no characters,", "")

        meta = {
            "input_kind": "poetry_llm" if llm_has_content else "poetry_fallback",
            "poem_raw": poem_raw,
            "layers": layers,
            "llm_raw": llm_raw,
            "llm_error": llm_error
        }

        return PromptBundle(
            global_prompt=global_style,
            final_prompt=final_prompt,
            negative_prompt=neg,
            meta=meta,
        )

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
            (["雪"], "snow"),
            (["孤舟", "舟", "船"], "small boat"),
            (["翁", "老"], "old man"),
            (["钓"], "fishing"),
        ]

        out: list[str] = []
        for keys, tag in rules:
            if any(k in text for k in keys) and tag not in out:
                out.append(tag)

        return out

    def _salvage_scene_from_raw(self, raw_response: str) -> str:
        # Deprecated: No longer needed with structured JSON
        return ""

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
