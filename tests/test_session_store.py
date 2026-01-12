from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from sdxl_app.storage.session_store import SessionManager


def test_session_create_save_history_revert():
    with TemporaryDirectory() as td:
        sessions_dir = Path(td) / "sessions"
        sm = SessionManager(sessions_dir)

        meta = sm.create()
        assert meta.session_id
        assert meta.current_version == -1
        assert meta.total_versions == 0

        img0 = Image.new("RGB", (64, 64), (10, 20, 30))
        v0 = sm.save_version(meta.session_id, img0, {"k": "v0"}, edit_type="generate", edit_text="scene")
        assert v0.version == 0
        assert v0.image_path.exists()
        assert v0.thumb_path.exists()

        meta2 = sm.get_meta(meta.session_id)
        assert meta2.current_version == 0
        assert meta2.total_versions == 1

        img1 = Image.new("RGB", (64, 64), (30, 20, 10))
        v1 = sm.save_version(meta.session_id, img1, {"k": "v1"}, edit_type="edit", edit_text="edit")
        assert v1.version == 1

        history = sm.history(meta.session_id)
        assert [it.version for it in history] == [0, 1]

        sm.revert(meta.session_id, 0)
        meta3 = sm.get_meta(meta.session_id)
        assert meta3.current_version == 0

        current = sm.current_image(meta.session_id)
        assert current is not None
        assert current.size == (64, 64)
