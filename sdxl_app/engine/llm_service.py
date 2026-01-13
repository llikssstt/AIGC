# ============================================================
# LLM Service - 本地大模型古诗理解服务
# ============================================================
"""
使用本地 LLM（通过 OpenAI 兼容 API，如 vLLM）来理解古诗并生成 SDXL prompt。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import json
import re
import httpx

logger = logging.getLogger(__name__)


@dataclass
class PoetryInterpretation:
    """LLM 对古诗的解读结果"""
    scene_description: str  # 场景描述（英文）
    visual_elements: list[str]  # 视觉元素标签
    mood: str  # 氛围/情绪
    time_of_day: Optional[str] = None  # 时间
    season: Optional[str] = None  # 季节
    raw_response: str = ""  # 原始 LLM 响应


# 古诗理解系统提示词 - 精简版
POETRY_SYSTEM_PROMPT = """You are an expert in classical Chinese poetry. Interpret the poem and describe its visual scene for image generation.

Output format (keep each line SHORT):
SCENE: [One sentence, max 15 words, describing the main visual scene]
ELEMENTS: [5-8 key visual elements, comma-separated, 2-3 words each]
MOOD: [1-2 mood words]
TIME: [time of day or "none"]
SEASON: [season or "none"]

Example for "床前明月光":
SCENE: poet gazing at moonlight on floor
ELEMENTS: bright moon, moonlight, indoor scene, contemplative figure, night
MOOD: nostalgic
TIME: night
SEASON: autumn

IMPORTANT: Keep prompts SHORT. SDXL works best with concise descriptions."""


class LLMService:
    """本地 LLM 服务，用于古诗理解和翻译"""

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen3-1.7B",
        api_key: str = "EMPTY",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def interpret_poetry(self, poem: str, style: str = "水墨") -> Optional[PoetryInterpretation]:
        """
        使用 LLM 理解古诗，返回场景描述和视觉元素。
        
        Args:
            poem: 古诗文本
            style: 绘画风格（水墨/工笔/青绿）
            
        Returns:
            PoetryInterpretation 或 None（如果调用失败）
        """
        if not poem or not poem.strip():
            return None

        user_prompt = f"Style: {style}\nPoem:\n{poem}"

        try:
            response = self._chat_completion(
                system_prompt=POETRY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            if response:
                return self._parse_poetry_response(response)
        except Exception as e:
            logger.warning("LLM 古诗理解失败: %s", e)
        
        return None

    def translate_edit(self, edit_text: str) -> Optional[str]:
        """
        使用 LLM 将中文编辑指令翻译为英文 prompt。
        
        Args:
            edit_text: 中文编辑指令，如 "将亭子换成山"
            
        Returns:
            英文 prompt 或 None
        """
        if not edit_text or not edit_text.strip():
            return None

        system_prompt = """You are a translator for image editing prompts. Translate the Chinese editing instruction to a concise English description suitable for image inpainting.

Rules:
- For replacement (换成/改成): describe ONLY what should appear, not what to remove
- For deletion (删除/去掉): describe "empty area, natural background continuation"
- For addition (添加/加上): describe what to add
- Keep it concise, 5-15 words
- Use visual descriptive language

Examples:
"将亭子换成山" -> "tall misty mountain with dramatic peaks"
"去掉这棵树" -> "empty area, natural background continuation"
"添加一只仙鹤" -> "elegant crane in flight"
"""
        try:
            response = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=edit_text,
            )
            if response:
                return response.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning("LLM 翻译编辑指令失败: %s", e)
        
        return None

    def _chat_completion(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """调用 OpenAI 兼容的 chat completion API"""
        client = self._get_client()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 512,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = f"{self.base_url}/chat/completions"
        logger.debug("LLM 请求: %s", url)
        
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None

    def _parse_poetry_response(self, response: str) -> PoetryInterpretation:
        """解析 LLM 的古诗理解响应"""
        raw = (response or "").strip()
        if not raw:
            return PoetryInterpretation(
                scene_description="",
                visual_elements=[],
                mood="",
                time_of_day=None,
                season=None,
                raw_response=response or "",
            )

        # 兼容 JSON 输出（部分模型倾向直接输出对象）
        if raw.startswith("{") and raw.endswith("}"):
            try:
                obj = json.loads(raw)
                scene = (obj.get("scene") or obj.get("SCENE") or "").strip()
                elements_raw = obj.get("elements") or obj.get("ELEMENTS") or []
                if isinstance(elements_raw, str):
                    elements = [e.strip() for e in elements_raw.split(",") if e.strip()]
                elif isinstance(elements_raw, list):
                    elements = [str(e).strip() for e in elements_raw if str(e).strip()]
                else:
                    elements = []
                mood = (obj.get("mood") or obj.get("MOOD") or "").strip()
                time_raw = (obj.get("time") or obj.get("TIME") or "").strip().lower()
                season_raw = (obj.get("season") or obj.get("SEASON") or "").strip().lower()
                time_of_day = None if time_raw in ("", "none", "null") else time_raw
                season = None if season_raw in ("", "none", "null") else season_raw
                return PoetryInterpretation(
                    scene_description=scene,
                    visual_elements=elements,
                    mood=mood,
                    time_of_day=time_of_day,
                    season=season,
                    raw_response=response,
                )
            except Exception:
                pass

        lines = raw.split("\n")
        
        scene = ""
        elements: list[str] = []
        mood = ""
        time_of_day = None
        season = None

        kv_re = re.compile(r"^(scene|elements|mood|time|season)\\s*[:：\\-]\\s*(.*)$", re.IGNORECASE)
        
        for line in lines:
            line = line.strip()

            m = kv_re.match(line)
            if not m:
                continue

            key = m.group(1).lower()
            val = m.group(2).strip()
            if key == "scene":
                scene = val
            elif key == "elements":
                elements = [e.strip() for e in val.split(",") if e.strip()]
            elif key == "mood":
                mood = val
            elif key == "time":
                v = val.lower()
                time_of_day = None if v in ("none", "null", "") else v
            elif key == "season":
                v = val.lower()
                season = None if v in ("none", "null", "") else v
        
        return PoetryInterpretation(
            scene_description=scene,
            visual_elements=elements,
            mood=mood,
            time_of_day=time_of_day,
            season=season,
            raw_response=response,
        )

    def health_check(self) -> bool:
        """检查 LLM 服务是否可用"""
        try:
            client = self._get_client()
            resp = client.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client:
            self._client.close()
            self._client = None


# 全局单例（可选）
_llm_service: Optional[LLMService] = None


def get_llm_service(
    base_url: str = "http://localhost:8001/v1",
    model: str = "Qwen3-1.7B",
    **kwargs,
) -> LLMService:
    """获取或创建 LLM 服务单例"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService(base_url=base_url, model=model, **kwargs)
    return _llm_service
