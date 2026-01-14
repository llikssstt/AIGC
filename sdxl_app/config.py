from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal

from pydantic import BaseModel, Field


# Shortened to ~25 tokens each to leave room for scene content (CLIP max: 77 tokens)
DEFAULT_STYLE_PRESETS: Dict[str, str] = {
    "水墨": (
        "Chinese ink wash painting, shuimo, flowing brushwork, "
        "monochrome, misty, rice paper texture, masterpiece"
    ),
    "工笔": (
        "Chinese gongbi painting, fine brushwork, rich colors, "
        "silk texture, detailed, masterpiece"
    ),
    "青绿": (
        "Chinese qinglv landscape, blue-green, mineral pigments, "
        "Tang dynasty style, layered mountains, masterpiece"
    ),
}

DEFAULT_NEGATIVE_PROMPT: str = (
    "watermark, text, logo, signature, username, "
    "low quality, worst quality, blurry, pixelated, "
    "deformed, ugly, bad anatomy, extra limbs, "
    "photorealistic, 3d render, photography, "
    "modern elements, western style, "
    "nsfw, nude"
)

INPAINT_NEGATIVE_APPEND: str = (
    ", change background, outside mask changes, "
    "extra objects outside region, affect unmasked area, "
    "inconsistent style, different lighting"
)


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class UISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False


class ModelSettings(BaseModel):
    base_path: str = "models/stable-diffusion-xl-base-1.0"
    inpaint_path: str = "models/stable-diffusion-xl-1.0-inpainting-0.1"
    refiner_path: Optional[str] = None


class RuntimeSettings(BaseModel):
    device: Literal["cuda", "cpu"] = "cuda"
    dtype: Literal["fp16", "fp32"] = "fp16"

    enable_xformers: bool = True
    enable_cpu_offload: bool = True
    enable_vae_slicing: bool = False
    enable_vae_tiling: bool = False


class DefaultsSettings(BaseModel):
    generate_steps: int = 30
    generate_cfg: float = 7.5
    generate_height: int = 1024
    generate_width: int = 1024

    edit_steps: int = 30
    edit_cfg: float = 7.5
    edit_strength: float = 0.6
    edit_grow_pixels: int = 8
    edit_blur_sigma: float = 12.0
    edit_invert_mask: bool = False


class StorageSettings(BaseModel):
    sessions_dir: Path = Field(default_factory=lambda: Path("storage/sessions"))


class PromptSettings(BaseModel):
    style_presets: Dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_STYLE_PRESETS))
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    inpaint_negative_append: str = INPAINT_NEGATIVE_APPEND

    # Poetry mode (for Chinese poem / lyrical input)
    poetry_enabled: bool = True
    poetry_negative_append: str = ", calligraphy, chinese characters, letters, words, subtitles"

    # LLM 配置（用于古诗理解）
    llm_enabled: bool = True  # 启用 LLM 增强
    llm_base_url: str = "http://localhost:8001/v1"  # vLLM OpenAI 兼容端点
    llm_model: str = "Qwen3-1.7B"  # 模型名称
    llm_api_key: str = "EMPTY"  # vLLM 不需要真实 key
    llm_timeout: float = 30.0  # 超时秒数


class DownloadSettings(BaseModel):
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    revision: Optional[str] = None
    fp16_only: bool = True


class Settings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    ui: UISettings = Field(default_factory=UISettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    defaults: DefaultsSettings = Field(default_factory=DefaultsSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)

    log_level: str = "INFO"
    config_file: Optional[Path] = None

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def api_url(self) -> str:
        return f"http://{self.server.host}:{self.server.port}"

    def ensure_dirs(self) -> None:
        self.storage.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.download.models_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, config_file: Optional[str | Path] = None, env_prefix: str = "SDXL_") -> "Settings":
        data: Dict[str, Any] = {}

        cfg_env = os.environ.get(f"{env_prefix}CONFIG")
        cfg_path = Path(config_file or cfg_env) if (config_file or cfg_env) else None
        if cfg_path:
            cfg_path = cfg_path.expanduser()
            if cfg_path.is_file():
                data = _load_yaml(cfg_path)
                data["config_file"] = cfg_path

        _apply_env_overrides(data, env_prefix=env_prefix)
        settings = cls.model_validate(data)

        # 统一把相对路径解析到项目根，避免 cwd 不同导致读错
        settings.storage.sessions_dir = _resolve_path(settings.project_root, settings.storage.sessions_dir)
        settings.download.models_dir = _resolve_path(settings.project_root, settings.download.models_dir)
        settings.ensure_dirs()
        return settings


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
        configure_logging(_settings)
    return _settings


def _resolve_path(base: Path, p: Path) -> Path:
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("YAML 配置需要安装 PyYAML: pip install pyyaml") from e

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return raw or {}


def _deep_set(d: Dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_env_overrides(data: Dict[str, Any], env_prefix: str) -> None:
    env_map: Dict[str, tuple[str, ...]] = {
        "SERVER_HOST": ("server", "host"),
        "SERVER_PORT": ("server", "port"),
        "UI_HOST": ("ui", "host"),
        "UI_PORT": ("ui", "port"),
        "MODELS_BASE_PATH": ("models", "base_path"),
        "MODELS_INPAINT_PATH": ("models", "inpaint_path"),
        "MODELS_REFINER_PATH": ("models", "refiner_path"),
        "RUNTIME_DEVICE": ("runtime", "device"),
        "RUNTIME_DTYPE": ("runtime", "dtype"),
        "RUNTIME_ENABLE_XFORMERS": ("runtime", "enable_xformers"),
        "RUNTIME_ENABLE_CPU_OFFLOAD": ("runtime", "enable_cpu_offload"),
        "RUNTIME_ENABLE_VAE_SLICING": ("runtime", "enable_vae_slicing"),
        "RUNTIME_ENABLE_VAE_TILING": ("runtime", "enable_vae_tiling"),
        "SESSIONS_DIR": ("storage", "sessions_dir"),
        "MODELS_DIR": ("download", "models_dir"),
        "PROMPTS_POETRY_ENABLED": ("prompts", "poetry_enabled"),
        "PROMPTS_POETRY_PREAMBLE": ("prompts", "poetry_preamble"),
        "PROMPTS_POETRY_NEGATIVE_APPEND": ("prompts", "poetry_negative_append"),
        "LOG_LEVEL": ("log_level",),
    }

    for suffix, path in env_map.items():
        name = f"{env_prefix}{suffix}"
        if name not in os.environ:
            continue
        _deep_set(data, path, os.environ[name])
