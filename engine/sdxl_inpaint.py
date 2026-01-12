# ============================================================
# SDXL Inpainting Editor - SDXL Inference Engine
# ============================================================
"""
SDXL 推理引擎模块：
- 文生图 Pipeline (StableDiffusionXLPipeline)
- Inpainting Pipeline (StableDiffusionXLInpaintPipeline)
- 可选 Refiner
"""
import logging
import gc
from typing import Optional, Tuple
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SDXLEngine:
    """
    SDXL 推理引擎
    
    支持两种模式：
    1. 文生图 (text2img): 生成初始图像 v0
    2. Inpainting: 基于 mask 的局部编辑
    """
    
    def __init__(
        self,
        text2img_path: str,
        inpaint_path: str,
        device: str = "cuda",
        torch_dtype: str = "float16",
        refiner_path: Optional[str] = None
    ):
        """
        初始化 SDXL 引擎
        
        Args:
            text2img_path: 文生图模型路径（HuggingFace ID 或本地路径）
            inpaint_path: Inpainting 模型路径
            device: 推理设备 ("cuda" 或 "cpu")
            torch_dtype: 精度 ("float16" 或 "float32")
            refiner_path: 可选 Refiner 模型路径
        """
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.text2img_path = text2img_path
        self.inpaint_path = inpaint_path
        self.refiner_path = refiner_path
        
        # 延迟加载 pipelines
        self._text2img_pipe = None
        self._inpaint_pipe = None
        self._refiner_pipe = None
        
        logger.info(f"SDXLEngine initialized: device={device}, dtype={torch_dtype}")
        logger.info(f"Text2Img model: {text2img_path}")
        logger.info(f"Inpaint model: {inpaint_path}")
        if refiner_path:
            logger.info(f"Refiner model: {refiner_path}")
    
    def _load_text2img_pipeline(self):
        """延迟加载文生图 Pipeline，加载前卸载其他模型以节省显存"""
        if self._text2img_pipe is not None:
            return
        
        # 卸载其他模型以节省显存
        self._unload_other_models(keep="text2img")
        
        logger.info("Loading Text2Img Pipeline...")
        
        try:
            from diffusers import AutoPipelineForText2Image
            
            self._text2img_pipe = AutoPipelineForText2Image.from_pretrained(
                self.text2img_path,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )
            self._text2img_pipe.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                self._text2img_pipe.enable_model_cpu_offload()
                try:
                    self._text2img_pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers attention enabled for text2img")
                except Exception as e:
                    logger.warning(f"xformers not available: {e}")
            
            logger.info("Text2Img Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Text2Img Pipeline: {e}")
            raise RuntimeError(f"无法加载文生图模型 '{self.text2img_path}': {e}")
    
    def _load_inpaint_pipeline(self):
        """延迟加载 Inpainting Pipeline，加载前卸载其他模型以节省显存"""
        if self._inpaint_pipe is not None:
            return
        
        # 卸载其他模型以节省显存
        self._unload_other_models(keep="inpaint")
        
        logger.info("Loading Inpaint Pipeline...")
        
        try:
            from diffusers import AutoPipelineForInpainting
            
            self._inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                self.inpaint_path,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )
            self._inpaint_pipe.to(self.device)
            
            # 启用内存优化
            if self.device == "cuda":
                self._inpaint_pipe.enable_model_cpu_offload()
                try:
                    self._inpaint_pipe.enable_xformers_memory_efficient_attention()
                    logger.info("xformers attention enabled for inpaint")
                except Exception as e:
                    logger.warning(f"xformers not available: {e}")
            
            logger.info("Inpaint Pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Inpaint Pipeline: {e}")
            raise RuntimeError(f"无法加载 Inpainting 模型 '{self.inpaint_path}': {e}")
    
    def _load_refiner_pipeline(self):
        """延迟加载 Refiner Pipeline"""
        if self._refiner_pipe is not None or self.refiner_path is None:
            return
        
        logger.info("Loading Refiner Pipeline...")
        
        try:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            
            self._refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.refiner_path,
                torch_dtype=self.torch_dtype,
                use_safetensors=True,
                variant="fp16" if self.torch_dtype == torch.float16 else None,
            )
            self._refiner_pipe.to(self.device)
            
            if self.device == "cuda":
                self._refiner_pipe.enable_model_cpu_offload()
            
            logger.info("Refiner Pipeline loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load Refiner Pipeline: {e}")
            self._refiner_pipe = None
    
    def _unload_other_models(self, keep: str = None):
        """
        卸载其他模型以节省显存（互斥加载策略）
        
        Args:
            keep: 要保留的模型 ("text2img", "inpaint", "refiner")
        """
        unloaded = []
        
        if keep != "text2img" and self._text2img_pipe is not None:
            del self._text2img_pipe
            self._text2img_pipe = None
            unloaded.append("text2img")
        
        if keep != "inpaint" and self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
            unloaded.append("inpaint")
        
        if keep != "refiner" and self._refiner_pipe is not None:
            del self._refiner_pipe
            self._refiner_pipe = None
            unloaded.append("refiner")
        
        if unloaded:
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"卸载模型以节省显存: {', '.join(unloaded)}")
    
    def _get_generator(self, seed: int) -> torch.Generator:
        """获取随机数生成器"""
        generator = torch.Generator(device=self.device)
        if seed >= 0:
            generator.manual_seed(seed)
            logger.debug(f"Using seed: {seed}")
        else:
            # 随机 seed
            seed = generator.seed()
            logger.debug(f"Random seed: {seed}")
        return generator, seed
    
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int = -1,
        steps: int = 30,
        cfg: float = 7.5,
        height: int = 1024,
        width: int = 1024,
        use_refiner: bool = False
    ) -> Tuple[Image.Image, int]:
        """
        文生图生成
        
        Args:
            prompt: 正向提示词
            negative_prompt: 负面提示词
            seed: 随机种子（-1 为随机）
            steps: 推理步数
            cfg: CFG Scale
            height: 图像高度
            width: 图像宽度
            use_refiner: 是否使用 Refiner
            
        Returns:
            Tuple of (生成的图像, 实际使用的 seed)
        """
        logger.info(f"Generating image: {width}x{height}, steps={steps}, cfg={cfg}")
        
        # 加载 pipeline
        self._load_text2img_pipeline()
        
        # 设置随机种子
        generator, actual_seed = self._get_generator(seed)
        
        try:
            # 生成图像
            result = self._text2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=generator,
            )
            
            image = result.images[0]
            
            # 可选：使用 Refiner 精修
            if use_refiner and self.refiner_path:
                self._load_refiner_pipeline()
                if self._refiner_pipe is not None:
                    logger.info("Applying Refiner...")
                    refined = self._refiner_pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        image=image,
                        num_inference_steps=steps // 2,
                        guidance_scale=cfg,
                        generator=generator,
                    )
                    image = refined.images[0]
            
            logger.info(f"Generation complete, seed={actual_seed}")
            return image, actual_seed
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"图像生成失败: {e}")
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str,
        seed: int = -1,
        steps: int = 30,
        cfg: float = 7.5,
        strength: float = 0.6
    ) -> Tuple[Image.Image, int]:
        """
        Inpainting 局部编辑
        
        Args:
            image: 原始图像
            mask: 编辑 mask（白色=编辑区域）
            prompt: 正向提示词
            negative_prompt: 负面提示词
            seed: 随机种子（-1 为随机）
            steps: 推理步数
            cfg: CFG Scale
            strength: 去噪强度 (0-1)
            
        Returns:
            Tuple of (编辑后的图像, 实际使用的 seed)
        """
        logger.info(f"Inpainting: steps={steps}, cfg={cfg}, strength={strength}")
        
        # 加载 pipeline
        self._load_inpaint_pipeline()
        
        # 确保图像和 mask 模式正确
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "L":
            mask = mask.convert("L")
        
        # 确保尺寸匹配
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        
        # 设置随机种子
        generator, actual_seed = self._get_generator(seed)
        
        try:
            # Inpainting
            result = self._inpaint_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                height=image.height,
                width=image.width,
                num_inference_steps=steps,
                guidance_scale=cfg,
                strength=strength,
                generator=generator,
            )
            
            edited_image = result.images[0]
            
            logger.info(f"Inpainting complete, seed={actual_seed}")
            return edited_image, actual_seed
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise RuntimeError(f"Inpainting 编辑失败: {e}")
    
    def unload_models(self):
        """卸载所有模型以释放显存"""
        logger.info("Unloading models...")
        
        if self._text2img_pipe is not None:
            del self._text2img_pipe
            self._text2img_pipe = None
        
        if self._inpaint_pipe is not None:
            del self._inpaint_pipe
            self._inpaint_pipe = None
        
        if self._refiner_pipe is not None:
            del self._refiner_pipe
            self._refiner_pipe = None
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("Models unloaded")
    
    def get_model_info(self) -> dict:
        """获取模型加载状态信息"""
        return {
            "text2img_loaded": self._text2img_pipe is not None,
            "inpaint_loaded": self._inpaint_pipe is not None,
            "refiner_loaded": self._refiner_pipe is not None,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "text2img_path": self.text2img_path,
            "inpaint_path": self.inpaint_path,
            "refiner_path": self.refiner_path,
        }


# ============================================================
# 全局引擎实例（单例模式）
# ============================================================
_engine_instance: Optional[SDXLEngine] = None


def get_engine() -> SDXLEngine:
    """获取全局 SDXL 引擎实例"""
    global _engine_instance
    
    if _engine_instance is None:
        # 导入配置
        try:
            from config import (
                MODEL_PATH_TEXT2IMG,
                MODEL_PATH_INPAINT,
                MODEL_PATH_REFINER,
                DEVICE,
                TORCH_DTYPE
            )
        except ImportError:
            # 使用默认配置
            MODEL_PATH_TEXT2IMG = "stabilityai/stable-diffusion-xl-base-1.0"
            MODEL_PATH_INPAINT = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
            MODEL_PATH_REFINER = None
            DEVICE = "cuda"
            TORCH_DTYPE = "float16"
        
        _engine_instance = SDXLEngine(
            text2img_path=MODEL_PATH_TEXT2IMG,
            inpaint_path=MODEL_PATH_INPAINT,
            device=DEVICE,
            torch_dtype=TORCH_DTYPE,
            refiner_path=MODEL_PATH_REFINER
        )
    
    return _engine_instance


def reset_engine():
    """重置全局引擎实例"""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.unload_models()
        _engine_instance = None
