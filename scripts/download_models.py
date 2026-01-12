#!/usr/bin/env python3
"""
SDXL 模型下载脚本 (FP16 专用版)

功能：
1. 仅下载 fp16 权重（*.fp16.safetensors）与必要配置文件
2. 严格排除非 fp16 大权重与 ONNX 文件
3. 支持 SDXL (base + inpaint) 与 DreamShaper 8 (text2img + inpaint)
4. 自动清理不需要的大文件（--clean）
5. 下载后进行 sanity check
6. 适配 Windows/Linux 路径

使用方法：
    # 下载 SDXL (默认)
    python scripts/download_models.py
    
    # 下载 DreamShaper 8 (轻量)
    python scripts/download_models.py --light
    
    # 下载前清理旧文件
    python scripts/download_models.py --clean
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from huggingface_hub import snapshot_download

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================
# 全局配置
# ============================================================
DOWNLOAD_ROOT = Path(__file__).parent.parent / "models"
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# 必须下载的文件模式（fp16 + 配置）
ALLOW_PATTERNS = [
    # 索引文件
    "model_index.json",
    
    # 所有 JSON 配置文件
    "**/*.json",
    
    # fp16 权重文件
    "**/*.fp16.safetensors",
    
    # Tokenizer 必需文件
    "**/*.txt",           # merges.txt, vocab.txt, special_tokens_map.txt 等
    "**/*.model",         # sentencepiece 模型（如 spiece.model）
]

# 必须排除的文件模式
IGNORE_PATTERNS = [
    # 非 fp16 的大权重文件
    "**/model.safetensors",
    "**/diffusion_pytorch_model.safetensors",
    "**/pytorch_model.bin",
    "**/diffusion_pytorch_model.bin",
    "**/*.ckpt",
    
    # ONNX 相关
    "**/*.onnx",
    "**/*.onnx_data",
    
    # 图片和文档
    "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.webp",
    "**/*.md", "**/LICENSE*", "**/.git*", "**/.gitattributes",
]

# ============================================================
# 模型配置
# ============================================================
MODELS_SDXL: List[Dict[str, Any]] = [
    {
        "name": "SDXL Base (fp16)",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "local_dir": DOWNLOAD_ROOT / "stable-diffusion-xl-base-1.0",
        "type": "text2img",
        "components": ["tokenizer", "tokenizer_2", "scheduler", "text_encoder", "text_encoder_2", "unet", "vae"],
    },
    {
        "name": "SDXL Inpainting (fp16)",
        "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "local_dir": DOWNLOAD_ROOT / "stable-diffusion-xl-1.0-inpainting-0.1",
        "type": "inpaint",
        "components": ["tokenizer", "tokenizer_2", "scheduler", "text_encoder", "text_encoder_2", "unet", "vae"],
    }
]

MODELS_LIGHT: List[Dict[str, Any]] = [
    {
        "name": "DreamShaper 8 (fp16)",
        "repo_id": "Lykon/dreamshaper-8",
        "local_dir": DOWNLOAD_ROOT / "dreamshaper-8",
        "type": "text2img",
        "components": ["tokenizer", "scheduler", "text_encoder", "unet", "vae", "feature_extractor", "safety_checker"],
    },
    {
        "name": "DreamShaper 8 Inpainting (fp16)",
        "repo_id": "Lykon/dreamshaper-8-inpainting",
        "local_dir": DOWNLOAD_ROOT / "dreamshaper-8-inpainting",
        "type": "inpaint",
        "components": ["tokenizer", "scheduler", "text_encoder", "unet", "vae", "feature_extractor", "safety_checker"],
    }
]

# ============================================================
# 清理函数
# ============================================================
def clean_unwanted_files(local_dir: Path) -> None:
    """
    删除本地目录中不需要的大文件（非 fp16 权重、ONNX 等）
    
    Args:
        local_dir: 模型本地目录
    """
    if not local_dir.exists():
        logger.info(f"⏭️  目录不存在，跳过清理: {local_dir}")
        return
    
    logger.info(f"🧹 开始清理目录: {local_dir}")
    
    unwanted_patterns = [
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
        "*.onnx",
        "*.onnx_data",
        "*.bin",
        "*.ckpt",
    ]
    
    deleted_count = 0
    deleted_size = 0
    
    for pattern in unwanted_patterns:
        for file_path in local_dir.rglob(pattern):
            if file_path.is_file():
                # 排除 fp16 文件
                if ".fp16." in file_path.name:
                    continue
                    
                file_size = file_path.stat().st_size
                try:
                    file_path.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                    logger.info(f"  🗑️  删除: {file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    logger.warning(f"  ⚠️  删除失败 {file_path.name}: {e}")
    
    if deleted_count > 0:
        logger.info(f"✅ 清理完成，删除 {deleted_count} 个文件，释放 {deleted_size / 1024 / 1024:.1f} MB")
    else:
        logger.info("✅ 无需清理")

# ============================================================
# 下载函数
# ============================================================
def download_model(model_config: Dict[str, Any]) -> bool:
    """
    下载单个模型
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        下载是否成功
    """
    name = model_config["name"]
    repo_id = model_config["repo_id"]
    local_dir = model_config["local_dir"]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📦 下载模型: {name}")
    logger.info(f"📍 仓库: {repo_id}")
    logger.info(f"💾 本地路径: {local_dir}")
    logger.info(f"{'='*60}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )
        logger.info(f"✅ 下载完成: {name}\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载失败 {name}: {e}\n")
        return False

# ============================================================
# Sanity Check
# ============================================================
def sanity_check(model_config: Dict[str, Any]) -> bool:
    """
    检查下载的模型是否完整
    
    Args:
        model_config: 模型配置字典
        
    Returns:
        检查是否通过
    """
    local_dir = model_config["local_dir"]
    components = model_config["components"]
    name = model_config["name"]
    
    logger.info(f"\n🔍 开始检查模型完整性: {name}")
    logger.info(f"📂 检查路径: {local_dir}\n")
    
    issues = []
    
    # 1. 检查 model_index.json
    model_index = local_dir / "model_index.json"
    if not model_index.exists():
        issues.append(f"❌ 缺少 model_index.json")
    else:
        logger.info(f"✅ model_index.json 存在")
    
    # 2. 检查各组件目录
    for component in components:
        component_dir = local_dir / component
        
        if not component_dir.exists():
            issues.append(f"❌ 缺少组件目录: {component}/")
            continue
        
        # 检查是否有 fp16 权重或配置文件
        has_fp16 = any(component_dir.rglob("*.fp16.safetensors"))
        has_config = (component_dir / "config.json").exists() or (component_dir / "tokenizer_config.json").exists()
        
        if has_fp16:
            logger.info(f"✅ {component}/ (包含 .fp16.safetensors)")
        elif has_config:
            logger.info(f"✅ {component}/ (包含配置文件)")
        else:
            # 特殊处理：scheduler 通常只有 JSON
            if component == "scheduler":
                logger.info(f"✅ {component}/ (scheduler 组件)")
            else:
                issues.append(f"⚠️  {component}/ 存在但缺少权重文件")
    
    # 3. 报告结果
    if issues:
        logger.warning(f"\n⚠️  检查发现 {len(issues)} 个问题:\n")
        for issue in issues:
            logger.warning(f"  {issue}")
        
        logger.warning(f"\n💡 建议：")
        logger.warning(f"  1. 检查网络连接后重新运行脚本（支持断点续传）")
        logger.warning(f"  2. 如果持续失败，可能需要调整 allow_patterns")
        logger.warning(f"  3. 手动检查 {local_dir} 目录内容\n")
        return False
    else:
        logger.info(f"\n✅ 模型完整性检查通过: {name}\n")
        return True

# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SDXL 模型下载脚本 (仅 fp16 权重)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/download_models.py              # 下载 SDXL
  python scripts/download_models.py --light      # 下载 DreamShaper 8
  python scripts/download_models.py --clean      # 下载前清理旧文件
"""
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="下载轻量级模型 DreamShaper 8 (~4GB) 而非 SDXL (~12GB)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="下载前清理已存在的非 fp16 大文件与 ONNX 文件"
    )
    
    args = parser.parse_args()
    
    # 选择模型列表
    if args.light:
        models = MODELS_LIGHT
        logger.info("💡 选择下载轻量级模型: DreamShaper 8")
    else:
        models = MODELS_SDXL
        logger.info("💎 选择下载 SDXL 模型")
        logger.info("提示: 如需轻量版，运行 `python scripts/download_models.py --light`")
    
    logger.info(f"📁 下载根目录: {DOWNLOAD_ROOT}\n")
    
    # 清理（如果指定）
    if args.clean:
        logger.info("🧹 执行清理模式...\n")
        for model in models:
            clean_unwanted_files(model["local_dir"])
        logger.info("")
    
    # 下载所有模型
    success_count = 0
    for model in models:
        if download_model(model):
            success_count += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 下载统计: {success_count}/{len(models)} 成功")
    logger.info(f"{'='*60}\n")
    
    # Sanity Check
    logger.info(f"\n{'='*60}")
    logger.info("🔍 开始完整性检查")
    logger.info(f"{'='*60}")
    
    check_passed = 0
    for model in models:
        if sanity_check(model):
            check_passed += 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 检查结果: {check_passed}/{len(models)} 通过")
    logger.info(f"{'='*60}\n")
    
    # 输出配置提示
    if check_passed == len(models):
        logger.info("\n🎉 所有模型下载并验证成功！\n")
        logger.info("【配置说明】推荐通过环境变量或 YAML 配置使用新结构 (sdxl_app/config.py):\n")

        text2img_model = next(m for m in models if m["type"] == "text2img")
        inpaint_model = next(m for m in models if m["type"] == "inpaint")

        logger.info("环境变量示例 (PowerShell):")
        logger.info(f'  $env:SDXL_MODELS_BASE_PATH="{text2img_model["local_dir"]}"')
        logger.info(f'  $env:SDXL_MODELS_INPAINT_PATH="{inpaint_model["local_dir"]}"')
        logger.info("")
        logger.info("或 YAML (例如 sdxl.yaml):")
        logger.info("  models:")
        logger.info(f'    base_path: "{text2img_model["local_dir"]}"')
        logger.info(f'    inpaint_path: "{inpaint_model["local_dir"]}"')
        logger.info("  runtime:")
        logger.info('    device: "cuda"')
        logger.info('    dtype: "fp16"')
        logger.info("然后运行:")
        logger.info("  set SDXL_CONFIG=sdxl.yaml  (Windows CMD)")
        logger.info("  # 或 PowerShell: $env:SDXL_CONFIG=\"sdxl.yaml\"")
        logger.info("")

        return 0
    else:
        logger.error("\n⚠️  部分模型检查未通过，请检查上述输出")
        return 1

if __name__ == "__main__":
    sys.exit(main())
