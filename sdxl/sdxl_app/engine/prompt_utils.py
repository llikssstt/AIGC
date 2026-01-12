from __future__ import annotations

import logging
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field

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
    ):
        self.style_presets = dict(style_presets)
        self.negative_prompt = negative_prompt
        self.inpaint_negative_append = inpaint_negative_append
        self.poetry_enabled = poetry_enabled
        self.poetry_preamble = poetry_preamble
        self.poetry_negative_append = poetry_negative_append

    def available_styles(self) -> list[str]:
        return sorted(self.style_presets.keys())

    def compile_generation(self, style_preset: str, scene_text: str) -> PromptBundle:
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        global_prompt = self.style_presets[style_preset].strip()
        scene_text = (scene_text or "").strip()

        if self.poetry_enabled and self._looks_like_poetry(scene_text):
            return self.compile_poetry(style_preset=style_preset, poem_text=scene_text)

        final_prompt = f"{global_prompt}, {scene_text}" if scene_text else global_prompt
        neg = self.negative_prompt
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta={})

    def compile_poetry(self, *, style_preset: str, poem_text: str) -> PromptBundle:
        """
        Poetry mode:
        - 不把原诗句直接塞进 prompt（降低“生成文字”的概率）
        - 从诗句中抽取意象/时间/季节/氛围，转成更 SDXL 友好的英文视觉标签
        """
        if style_preset not in self.style_presets:
            raise ValueError(f"Unknown style_preset: {style_preset}")

        global_prompt = self.style_presets[style_preset].strip()
        poem_raw = (poem_text or "").strip()
        poem_norm = _normalize_poem(poem_raw)

        tags, meta = _extract_poetry_meta(poem_norm)
        parts = [global_prompt, self.poetry_preamble]
        parts.extend(tags)
        final_prompt = ", ".join([p for p in parts if p])

        neg = self.negative_prompt + self.poetry_negative_append
        extra_neg = meta.get("poetry_negative_terms") if isinstance(meta, dict) else None
        if isinstance(extra_neg, list) and extra_neg:
            neg += ", " + ", ".join([str(x) for x in extra_neg if x])
        meta = {**meta, "input_kind": "poetry", "poem_raw": poem_raw}
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta=meta)

    def _looks_like_poetry(self, text: str) -> bool:
        if not text:
            return False
        # 多行/常见诗词标点/明显中文字符 → 进入诗词增强模式
        if "\n" in text:
            return True
        if any(p in text for p in ("，", "。", "；", "！", "？")):
            return _contains_cjk(text)
        # 纯中文短句也有可能是“诗意一句话”
        return _contains_cjk(text) and len(text) <= 80

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
        return PromptBundle(global_prompt=global_prompt, final_prompt=final_prompt, negative_prompt=neg, meta={})

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


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _normalize_poem(text: str) -> str:
    # 统一分隔符，去掉多余空白
    t = (text or "").strip()
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    return t


def _extract_poetry_meta(poem: str) -> tuple[list[str], dict[str, Any]]:
    # 简单、可维护的意象词典（可后续挪到 YAML 配置）
    lexicon = [
        ("床前", "bedside"),
        ("举杯", "raising a wine cup"),
        ("对影", "facing his shadow"),
        ("独酌", "drinking alone"),
        ("明月", "bright moon"),
        ("望月", "gazing at the moon"),
        ("月", "moon"),
        ("星", "stars"),
        ("霜", "frost"),
        ("雪", "snow"),
        ("雨", "rain"),
        ("风", "wind"),
        ("云", "clouds"),
        ("雾", "mist"),
        ("烟", "haze"),
        ("江", "river"),
        ("湖", "lake"),
        ("海", "sea"),
        ("山", "mountains"),
        ("松", "pine tree"),
        ("竹", "bamboo"),
        ("梅", "plum blossoms"),
        ("柳", "willow"),
        ("花", "flowers"),
        ("草", "grass"),
        ("舟", "boat"),
        ("孤舟", "lonely boat"),
        ("渔", "fisherman"),
        ("寺", "temple"),
        ("城", "ancient town"),
        ("桥", "stone bridge"),
        ("灯", "lantern"),
        ("酒", "wine"),
        ("杯", "wine cup"),
        ("影", "shadow"),
        ("三人", "three companions"),
        ("剑", "sword"),
        ("马", "horse"),
    ]
    lexicon.sort(key=lambda x: len(x[0]), reverse=True)

    imagery: list[str] = []
    matched: list[str] = []
    for zh, en in lexicon:
        if zh in poem:
            imagery.append(en)
            matched.append(zh)

    # 时间/季节/情绪：非常轻量的规则，避免引入 LLM
    time_tags: list[str] = []
    if any(x in poem for x in ("夜", "月", "星", "霜")):
        time_tags.append("night")
    if any(x in poem for x in ("晓", "晨", "朝", "日出")):
        time_tags.append("dawn")
    if any(x in poem for x in ("夕", "暮", "黄昏", "落日")):
        time_tags.append("sunset")

    season_tags: list[str] = []
    if "春" in poem:
        season_tags.append("spring")
    if "夏" in poem:
        season_tags.append("summer")
    if "秋" in poem:
        season_tags.append("autumn")
    if "冬" in poem:
        season_tags.append("winter")

    mood_tags: list[str] = []
    if any(x in poem for x in ("思", "乡", "故", "归")):
        mood_tags.append("nostalgic")
    if any(x in poem for x in ("愁", "泪", "伤", "别")):
        mood_tags.append("melancholic")
    if any(x in poem for x in ("孤", "寂")):
        mood_tags.append("lonely")
    if any(x in poem for x in ("静", "清")):
        mood_tags.append("tranquil")

    # 诗词增强：额外补充一些“叙事/主体”标签（比纯名词更容易让模型画对）
    has_moon = any(x in poem for x in ("月", "明月", "月光", "皓月"))
    has_wine = any(x in poem for x in ("酒", "杯", "樽", "酌", "饮", "醉", "举杯"))
    has_shadow = "影" in poem
    has_three = "三人" in poem or ("三" in poem and "人" in poem)
    has_person = any(x in poem for x in ("我", "君", "子", "翁", "客", "僧", "仙", "举杯", "独酌", "对影")) or has_wine

    narrative_tags: list[str] = []
    scene_phrases: list[str] = []
    if has_person:
        narrative_tags.extend(
            [
                "ancient Chinese poet",
                "traditional hanfu robe",
                "single human figure",
                "ink wash figure painting",
                "figure-focused composition",
                "full-body shot",
                "centered composition",
                "minimal background",
            ]
        )
    if has_wine:
        narrative_tags.append("toasting with a wine cup")
    if has_moon:
        narrative_tags.append("bright full moon")
        narrative_tags.append("moonlight")
    if has_shadow:
        narrative_tags.append("a clear shadow on the ground")
    if has_three:
        narrative_tags.append("three companions: the poet, the moon, and the shadow")

    # 场景倾向：没有明确地名时，给一个“中性且合理”的场景
    if has_person and has_moon and ("river" not in imagery) and ("mountains" not in imagery) and ("temple" not in imagery):
        narrative_tags.append("quiet night courtyard")
    # 月 + 酒 常见“浪漫/洒脱”氛围
    if has_moon and has_wine:
        mood_tags.append("romantic")
        mood_tags.append("contemplative")

    # 更“可画”的一句话场景描述（优先用于主体明确的诗句）
    if has_person and has_moon and has_wine:
        scene = "a lone ancient Chinese poet in hanfu raising a wine cup toward a bright full moon"
        if has_shadow:
            scene += ", his shadow clearly visible on the ground"
        if has_three:
            scene += ", symbolic three companions (poet, moon, shadow)"
        scene_phrases.append(scene)
    elif has_person and has_wine:
        scene_phrases.append("a lone ancient Chinese poet drinking wine alone")
    elif has_moon:
        scene_phrases.append("a bright full moon in the night sky, quiet and tranquil")

    # 去重并固定顺序
    def uniq(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for it in items:
            if it and it not in seen:
                seen.add(it)
                out.append(it)
        return out

    imagery = uniq(imagery)
    time_tags = uniq(time_tags)
    season_tags = uniq(season_tags)
    mood_tags = uniq(mood_tags)
    narrative_tags = uniq(narrative_tags)
    scene_phrases = uniq(scene_phrases)

    # 合成标签：先时间/季节/氛围，再意象
    tags: list[str] = []
    tags.extend(scene_phrases)
    tags.extend(narrative_tags)
    tags.extend(time_tags)
    tags.extend(season_tags)
    tags.extend(mood_tags)
    tags.extend(imagery)

    # 如果诗句主体明确但没提山水地景：适当压制“默认出山水”的倾向
    negative_terms: list[str] = []
    has_landscape = any(x in imagery for x in ("mountains", "river", "lake", "sea"))
    if has_person and not has_landscape:
        negative_terms.extend(
            [
                "landscape",
                "mountains",
                "river",
                "scenery",
                "wide shot",
                "empty landscape",
            ]
        )
    negative_terms = uniq(negative_terms)

    meta: dict[str, Any] = {
        "poetry_matched_zh": matched,
        "poetry_tags": tags,
        "poetry_scene": scene_phrases[0] if scene_phrases else None,
        "poetry_narrative": narrative_tags,
        "poetry_time": time_tags,
        "poetry_season": season_tags,
        "poetry_mood": mood_tags,
        "poetry_negative_terms": negative_terms,
    }
    return tags, meta
