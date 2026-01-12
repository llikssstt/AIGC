# ============================================================
# SDXL Inpainting Editor - Session Manager
# ============================================================
"""
Session 与版本管理模块：
- Session 创建与管理
- 版本历史记录
- 图像与参数存储
- 回退/撤销功能
"""
import logging
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VersionInfo:
    """版本信息"""
    version: int
    timestamp: str
    edit_type: str  # "generate" 或 "edit"
    edit_text: Optional[str]
    thumbnail_path: Optional[str]
    params: Dict[str, Any]


@dataclass
class SessionMetadata:
    """Session 元数据"""
    session_id: str
    created_at: str
    updated_at: str
    global_prompt: Optional[str]
    style_preset: Optional[str]
    current_version: int
    total_versions: int


class SessionManager:
    """
    Session 与版本管理器
    
    存储结构：
    storage/sessions/{session_id}/
    ├── metadata.json       # session 元数据
    ├── v0.png              # 版本 0 图片
    ├── v0_thumb.png        # 缩略图
    ├── v0_params.json      # 版本 0 参数
    ├── v1.png
    ├── v1_mask.png         # 编辑用的 mask
    ├── v1_thumb.png
    ├── v1_params.json
    └── ...
    """
    
    def __init__(self, storage_dir: Path):
        """
        初始化 Session 管理器
        
        Args:
            storage_dir: 存储根目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"SessionManager initialized: {self.storage_dir}")
    
    def _get_session_dir(self, session_id: str) -> Path:
        """获取 session 目录路径"""
        return self.storage_dir / session_id
    
    def _generate_thumbnail(self, image: Image.Image, size: tuple = (128, 128)) -> Image.Image:
        """生成缩略图"""
        thumb = image.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb
    
    def create_session(self) -> str:
        """
        创建新的 session
        
        Returns:
            session_id
        """
        session_id = str(uuid.uuid4())[:8]
        session_dir = self._get_session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化元数据
        metadata = SessionMetadata(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            global_prompt=None,
            style_preset=None,
            current_version=-1,
            total_versions=0
        )
        
        self._save_metadata(session_id, metadata)
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def session_exists(self, session_id: str) -> bool:
        """检查 session 是否存在"""
        return self._get_session_dir(session_id).exists()
    
    def _save_metadata(self, session_id: str, metadata: SessionMetadata):
        """保存 session 元数据"""
        metadata_path = self._get_session_dir(session_id) / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self, session_id: str) -> SessionMetadata:
        """加载 session 元数据"""
        metadata_path = self._get_session_dir(session_id) / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Session {session_id} not found")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return SessionMetadata(**data)
    
    def get_metadata(self, session_id: str) -> SessionMetadata:
        """获取 session 元数据"""
        return self._load_metadata(session_id)
    
    def set_global_prompt(self, session_id: str, global_prompt: str, style_preset: Optional[str] = None):
        """设置 session 的全局 prompt（在 generate 时调用）"""
        metadata = self._load_metadata(session_id)
        metadata.global_prompt = global_prompt
        metadata.style_preset = style_preset
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)
        logger.info(f"Session {session_id}: global_prompt set, style={style_preset}")
    
    def save_version(
        self,
        session_id: str,
        image: Image.Image,
        params: Dict[str, Any],
        edit_type: str = "edit",
        edit_text: Optional[str] = None,
        mask: Optional[Image.Image] = None
    ) -> int:
        """
        保存新版本
        
        Args:
            session_id: session ID
            image: 图像
            params: 参数字典
            edit_type: "generate" 或 "edit"
            edit_text: 编辑文本（仅 edit 类型）
            mask: 编辑 mask（仅 edit 类型）
            
        Returns:
            版本号
        """
        metadata = self._load_metadata(session_id)
        version = metadata.total_versions
        session_dir = self._get_session_dir(session_id)
        
        # 保存图像
        image_path = session_dir / f"v{version}.png"
        image.save(image_path, "PNG")
        
        # 保存缩略图
        thumb = self._generate_thumbnail(image)
        thumb_path = session_dir / f"v{version}_thumb.png"
        thumb.save(thumb_path, "PNG")
        
        # 保存 mask（如果有）
        if mask is not None:
            mask_path = session_dir / f"v{version}_mask.png"
            mask.save(mask_path, "PNG")
        
        # 保存参数
        version_params = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "edit_type": edit_type,
            "edit_text": edit_text,
            **params
        }
        params_path = session_dir / f"v{version}_params.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(version_params, f, ensure_ascii=False, indent=2)
        
        # 更新元数据
        metadata.current_version = version
        metadata.total_versions = version + 1
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)
        
        logger.info(f"Session {session_id}: saved version {version} ({edit_type})")
        return version
    
    def get_history(self, session_id: str) -> List[VersionInfo]:
        """
        获取版本历史列表
        
        Args:
            session_id: session ID
            
        Returns:
            版本信息列表
        """
        metadata = self._load_metadata(session_id)
        session_dir = self._get_session_dir(session_id)
        
        history = []
        for v in range(metadata.total_versions):
            params_path = session_dir / f"v{v}_params.json"
            if params_path.exists():
                with open(params_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                
                thumb_path = session_dir / f"v{v}_thumb.png"
                
                history.append(VersionInfo(
                    version=v,
                    timestamp=params.get("timestamp", ""),
                    edit_type=params.get("edit_type", "unknown"),
                    edit_text=params.get("edit_text"),
                    thumbnail_path=str(thumb_path) if thumb_path.exists() else None,
                    params=params
                ))
        
        return history
    
    def get_image(self, session_id: str, version: int) -> Image.Image:
        """
        获取指定版本的图像
        
        Args:
            session_id: session ID
            version: 版本号
            
        Returns:
            图像
        """
        session_dir = self._get_session_dir(session_id)
        image_path = session_dir / f"v{version}.png"
        
        if not image_path.exists():
            raise ValueError(f"Version {version} not found in session {session_id}")
        
        return Image.open(image_path)
    
    def get_thumbnail(self, session_id: str, version: int) -> Optional[Image.Image]:
        """获取指定版本的缩略图"""
        session_dir = self._get_session_dir(session_id)
        thumb_path = session_dir / f"v{version}_thumb.png"
        
        if thumb_path.exists():
            return Image.open(thumb_path)
        return None
    
    def get_mask(self, session_id: str, version: int) -> Optional[Image.Image]:
        """获取指定版本的 mask"""
        session_dir = self._get_session_dir(session_id)
        mask_path = session_dir / f"v{version}_mask.png"
        
        if mask_path.exists():
            return Image.open(mask_path)
        return None
    
    def get_params(self, session_id: str, version: int) -> Dict[str, Any]:
        """获取指定版本的参数"""
        session_dir = self._get_session_dir(session_id)
        params_path = session_dir / f"v{version}_params.json"
        
        if not params_path.exists():
            raise ValueError(f"Params for version {version} not found")
        
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def revert_to_version(self, session_id: str, version: int) -> Image.Image:
        """
        回退到指定版本（不删除后续版本，只更新 current_version）
        
        Args:
            session_id: session ID
            version: 目标版本号
            
        Returns:
            该版本的图像
        """
        metadata = self._load_metadata(session_id)
        
        if version < 0 or version >= metadata.total_versions:
            raise ValueError(f"Invalid version {version}, valid range: 0-{metadata.total_versions - 1}")
        
        # 更新当前版本指针
        metadata.current_version = version
        metadata.updated_at = datetime.now().isoformat()
        self._save_metadata(session_id, metadata)
        
        logger.info(f"Session {session_id}: reverted to version {version}")
        return self.get_image(session_id, version)
    
    def get_current_image(self, session_id: str) -> Optional[Image.Image]:
        """获取当前版本的图像"""
        metadata = self._load_metadata(session_id)
        if metadata.current_version < 0:
            return None
        return self.get_image(session_id, metadata.current_version)
    
    def get_current_version(self, session_id: str) -> int:
        """获取当前版本号"""
        metadata = self._load_metadata(session_id)
        return metadata.current_version
    
    def delete_session(self, session_id: str):
        """删除 session"""
        session_dir = self._get_session_dir(session_id)
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Deleted session: {session_id}")
    
    def list_sessions(self) -> List[str]:
        """列出所有 session"""
        sessions = []
        for d in self.storage_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                sessions.append(d.name)
        return sessions
    
    def export_image(self, session_id: str, version: int, export_path: str):
        """导出指定版本的图像到指定路径"""
        image = self.get_image(session_id, version)
        image.save(export_path)
        logger.info(f"Exported v{version} to {export_path}")


# ============================================================
# 全局实例
# ============================================================
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 实例"""
    global _session_manager
    
    if _session_manager is None:
        try:
            from config import STORAGE_DIR
        except ImportError:
            STORAGE_DIR = Path(__file__).parent.parent / "storage" / "sessions"
        
        _session_manager = SessionManager(STORAGE_DIR)
    
    return _session_manager
