# ============================================================
# LLM Service - 本地大模型古诗理解服务
# ============================================================
"""
使用本地 LLM（通过 OpenAI 兼容 API，如 vLLM）来理解古诗并生成 SDXL prompt。
Refactored: 输出结构化 JSON，明确区分 Subject/Action/Environment/Composition。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import json
import re
import httpx

logger = logging.getLogger(__name__)


@dataclass
class PoetryInterpretation:
    """LLM 对古诗的解读结果 (Structured)"""
    # 核心字段
    subject_description: str  # 主体描述 (e.g. "1 elderly man, long beard")
    action_description: str   # 动作 (e.g. "sitting alone, fishing")
    environment_description: str # 环境 (e.g. "snowy river, falling snow")
    composition_description: str # 构图 (e.g. "wide shot, negative space")
    mood_description: str       # 氛围 (e.g. "solitary, zen")
    
    # 辅助/元数据
    visual_elements: List[str]  # 补充的视觉标签
    raw_response: str = ""      # 原始响应


# 古诗理解系统提示词 - 结构化 JSON 版
POETRY_SYSTEM_PROMPT = """You are an expert in classical Chinese poetry and Prompt Engineering for SDXL. 
Your task is to interpret the user's poem and convert it into a STRUCTURED JSON format for image generation.

CRITICAL: You MUST output valid JSON. No markdown, no conversational text.

Structure your analysis into these SPECIFIC layers:
1. "subject": MAIN CHARACTER or OBJECT. Be explicit (e.g., "1 man", "old fisherman"). If implied, extract it. 
2. "action": What the subject is doing (e.g., "fishing", "drinking wine").
3. "environment": The setting, background, weather (e.g., "mountain river", "snowing").
4. "composition": Camera angle, framing (e.g., "wide shot", "close up on face", "negative space").
5. "mood": Emotional atmosphere (e.g., "lonely", "majestic").
6. "elements": A list of 10-15 short visual tags.

Example Input: "孤舟蓑笠翁，独钓寒江雪" (Lone boat, bamboo hat old man, fishing alone in cold river snow)
Example Output JSON:
{
  "subject": "1 elderly man, long white beard, wearing traditional straw raincoat and bamboo conical hat",
  "action": "sitting alone on a small wooden boat, holding a fishing rod, fishing",
  "environment": "vast river, frozen water, heavy falling snow, misty mountains in background",
  "composition": "wide shot, minimalist composition, large negative space, center composition",
  "mood": "solitary, peaceful, cold, zen",
  "elements": ["old man", "snowy river", "fishing boat", "falling snow", "mist"]
}

Example Input: "空山不见人" (Empty mountain, no one seen)
Example Output JSON:
{
  "subject": "no humans", 
  "action": "none",
  "environment": "dense deep forest, ancient trees, mossy rocks, shafts of sunlight, empty mountain path",
  "composition": "low angle, looking up at trees, depth of field",
  "mood": "quiet, mysterious, seclusive",
  "elements": ["forest", "trees", "light rays", "moss"]
}
"""


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
            # Avoid system proxy interference with local LLM server.
            self._client = httpx.Client(timeout=self.timeout, trust_env=False)
        return self._client

    def interpret_poetry(self, poem: str, style: str = "水墨") -> Optional[PoetryInterpretation]:
        """
        使用 LLM 理解古诗，返回结构化 Prompt 方案。
        """
        if not poem or not poem.strip():
            return None

        user_prompt = f"Style: {style}\nPoem: {poem}"

        try:
            response = self._chat_completion(
                system_prompt=POETRY_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            if response:
                return self._parse_json_response(response)
        except Exception as e:
            logger.warning("LLM 古诗理解失败: %s", e)
            raise e # Raise it so PromptCompiler knows it failed
        
        return None

    def translate_edit(self, edit_text: str) -> Optional[str]:
        """
        使用 LLM 将中文编辑指令翻译为英文 prompt。
        """
        if not edit_text or not edit_text.strip():
            return None

        system_prompt = """You are a translator for image editing prompts. Translate the Chinese editing instruction to a concise English description suitable for image inpainting.
        
        Rules:
        - For replacement (换成/改成): describe ONLY what should appear, not what to remove.
        - For deletion (删除/去掉): describe "empty area, natural background continuation".
        - For addition (添加/加上): describe what to add.
        - Keep it concise, 5-15 words.
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
            "temperature": 0.3, # Lower temperature for stable JSON
            "max_tokens": 512,
            "response_format": {"type": "json_object"}, # Hint for JSON-capable models
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        url = f"{self.base_url}/chat/completions"
        logger.debug("LLM 请求: %s", url)
        
        # 尝试标准请求
        try:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            # 如果 backend 不支持 response_format (400 error)，移除该字段重试
            if e.response.status_code == 400 and "response_format" in str(e.response.text):
                logger.warning("Model backend does not support 'response_format', retrying without it.")
                del payload["response_format"]
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
            else:
                raise e
        
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return None

    def _parse_json_response(self, response: str) -> PoetryInterpretation:
        """解析 LLM 返回的 JSON"""
        raw = (response or "").strip()
        
        # 提取 JSON 块
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = raw

        data = {}
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("LLM 返回了无效 JSON，尝试修复或降级: %s", raw)
            # 简单的降级逻辑：如果解析失败，把整个 raw 当作 subject (fallback)
            pass

        # 归一化字段 (处理大小写/缺失)
        def get_field(keys: List[str], default: str = "") -> str:
            for k in keys:
                if k in data and data[k]:
                    return str(data[k]).strip()
                if k.upper() in data and data[k.upper()]:
                    return str(data[k.upper()]).strip()
            return default

        subject = get_field(["subject", "main_character"], "")
        action = get_field(["action", "activity"], "")
        environment = get_field(["environment", "background"], "")
        composition = get_field(["composition", "framing"], "")
        mood = get_field(["mood", "atmosphere"], "")
        
        elements_raw = data.get("elements") or data.get("ELEMENTS") or []
        elements = []
        if isinstance(elements_raw, list):
            elements = [str(x) for x in elements_raw]
        elif isinstance(elements_raw, str):
            elements = [x.strip() for x in elements_raw.split(",")]

        return PoetryInterpretation(
            subject_description=subject,
            action_description=action,
            environment_description=environment,
            composition_description=composition,
            mood_description=mood,
            visual_elements=elements,
            raw_response=raw
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


# 全局单例
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
