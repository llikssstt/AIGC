from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionMeta:
    session_id: str
    created_at: str
    updated_at: str
    global_prompt: str
    style_preset: Optional[str]
    current_version: int
    total_versions: int


@dataclass(frozen=True)
class VersionItem:
    version: int
    timestamp: str
    edit_type: str
    edit_text: Optional[str]
    image_path: Path
    thumb_path: Path
    card: dict[str, Any]


class SessionManager:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create(self) -> SessionMeta:
        sid = uuid.uuid4().hex[:8]
        sdir = self._session_dir(sid)
        sdir.mkdir(parents=True, exist_ok=True)

        now = datetime.now().isoformat()
        meta = SessionMeta(
            session_id=sid,
            created_at=now,
            updated_at=now,
            global_prompt="",
            style_preset=None,
            current_version=-1,
            total_versions=0,
        )
        self._write_json(sdir / "metadata.json", _meta_to_dict(meta))
        return meta

    def exists(self, session_id: str) -> bool:
        return (self._session_dir(session_id) / "metadata.json").exists()

    def list_all(self) -> list[SessionMeta]:
        """List all sessions sorted by updated_at descending."""
        sessions = []
        for sdir in self.sessions_dir.iterdir():
            if sdir.is_dir():
                meta_path = sdir / "metadata.json"
                if meta_path.exists():
                    try:
                        data = self._read_json(meta_path)
                        sessions.append(_dict_to_meta(data))
                    except Exception as e:
                        logger.warning("Failed to read session %s: %s", sdir.name, e)
        # Sort by updated_at descending (newest first)
        sessions.sort(key=lambda m: m.updated_at, reverse=True)
        return sessions

    def get_meta(self, session_id: str) -> SessionMeta:
        data = self._read_json(self._session_dir(session_id) / "metadata.json")
        return _dict_to_meta(data)

    def set_global_prompt(self, session_id: str, global_prompt: str, style_preset: Optional[str]) -> SessionMeta:
        meta = self.get_meta(session_id)
        now = datetime.now().isoformat()
        meta2 = SessionMeta(
            session_id=meta.session_id,
            created_at=meta.created_at,
            updated_at=now,
            global_prompt=global_prompt,
            style_preset=style_preset,
            current_version=meta.current_version,
            total_versions=meta.total_versions,
        )
        self._write_json(self._session_dir(session_id) / "metadata.json", _meta_to_dict(meta2))
        return meta2

    def save_version(
        self,
        session_id: str,
        image: Image.Image,
        card: dict[str, Any],
        *,
        edit_type: str,
        edit_text: Optional[str] = None,
        mask: Optional[Image.Image] = None,
    ) -> VersionItem:
        meta = self.get_meta(session_id)
        version = meta.total_versions
        sdir = self._session_dir(session_id)

        image_path = sdir / f"v{version}.png"
        thumb_path = sdir / f"v{version}_thumb.png"
        card_path = sdir / f"v{version}_card.json"

        image.save(image_path, format="PNG")
        thumb = image.copy()
        thumb.thumbnail((256, 256), Image.Resampling.LANCZOS)
        thumb.save(thumb_path, format="PNG")

        if mask is not None:
            mask_path = sdir / f"v{version}_mask.png"
            mask.convert("L").save(mask_path, format="PNG")

        record = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "edit_type": edit_type,
            "edit_text": edit_text,
            "card": card,
        }
        self._write_json(card_path, record)

        now = datetime.now().isoformat()
        meta2 = SessionMeta(
            session_id=meta.session_id,
            created_at=meta.created_at,
            updated_at=now,
            global_prompt=meta.global_prompt,
            style_preset=meta.style_preset,
            current_version=version,
            total_versions=version + 1,
        )
        self._write_json(sdir / "metadata.json", _meta_to_dict(meta2))

        return VersionItem(
            version=version,
            timestamp=record["timestamp"],
            edit_type=edit_type,
            edit_text=edit_text,
            image_path=image_path,
            thumb_path=thumb_path,
            card=card,
        )

    def history(self, session_id: str) -> list[VersionItem]:
        meta = self.get_meta(session_id)
        sdir = self._session_dir(session_id)

        out: list[VersionItem] = []
        for v in range(meta.total_versions):
            card_path = sdir / f"v{v}_card.json"
            if not card_path.exists():
                continue
            record = self._read_json(card_path)
            out.append(
                VersionItem(
                    version=v,
                    timestamp=record.get("timestamp", ""),
                    edit_type=record.get("edit_type", "unknown"),
                    edit_text=record.get("edit_text"),
                    image_path=sdir / f"v{v}.png",
                    thumb_path=sdir / f"v{v}_thumb.png",
                    card=(record.get("card") or {}),
                )
            )
        return out

    def revert(self, session_id: str, version: int) -> SessionMeta:
        meta = self.get_meta(session_id)
        if version < 0 or version >= meta.total_versions:
            raise ValueError(f"Invalid version: {version}")

        now = datetime.now().isoformat()
        meta2 = SessionMeta(
            session_id=meta.session_id,
            created_at=meta.created_at,
            updated_at=now,
            global_prompt=meta.global_prompt,
            style_preset=meta.style_preset,
            current_version=version,
            total_versions=meta.total_versions,
        )
        self._write_json(self._session_dir(session_id) / "metadata.json", _meta_to_dict(meta2))
        return meta2

    def image_path(self, session_id: str, version: int) -> Path:
        return self._session_dir(session_id) / f"v{version}.png"

    def thumb_path(self, session_id: str, version: int) -> Path:
        return self._session_dir(session_id) / f"v{version}_thumb.png"

    def current_image(self, session_id: str) -> Optional[Image.Image]:
        meta = self.get_meta(session_id)
        if meta.current_version < 0:
            return None
        path = self.image_path(session_id, meta.current_version)
        with Image.open(path) as im:
            return im.copy()

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _read_json(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, obj: dict[str, Any]) -> None:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _meta_to_dict(m: SessionMeta) -> dict[str, Any]:
    return {
        "session_id": m.session_id,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "global_prompt": m.global_prompt,
        "style_preset": m.style_preset,
        "current_version": m.current_version,
        "total_versions": m.total_versions,
    }


def _dict_to_meta(d: dict[str, Any]) -> SessionMeta:
    return SessionMeta(
        session_id=d["session_id"],
        created_at=d["created_at"],
        updated_at=d["updated_at"],
        global_prompt=d.get("global_prompt", ""),
        style_preset=d.get("style_preset"),
        current_version=int(d.get("current_version", -1)),
        total_versions=int(d.get("total_versions", 0)),
    )

