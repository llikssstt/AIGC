# ============================================================
# SDXL Inpainting Editor - Evaluation Script
# ============================================================
"""
评估脚本：
计算编辑前后的各项指标，输出 CSV。
指标：
1. PSNR (非 mask 区域) - 越高越好，表示背景保持
2. SSIM (非 mask 区域) - 越高越好
3. Pixel Change Mean (mask 区域) - 表示编辑强度
"""
import sys
import argparse
import csv
import logging
import json
import re
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# 添加项目根目录到 path（便于直接运行脚本）
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


def calculate_metrics(original_path: str, edited_path: str, mask_path: str) -> dict:
    """计算图像指标"""
    # 1. 读取图像
    img_orig = cv2.imread(original_path)
    img_edit = cv2.imread(edited_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_orig is None or img_edit is None or mask is None:
        raise ValueError("Cannot read images")
    
    # 确保尺寸一致
    if img_orig.shape != img_edit.shape:
        img_edit = cv2.resize(img_edit, (img_orig.shape[1], img_orig.shape[0]))
    
    if mask.shape != img_orig.shape[:2]:
        mask = cv2.resize(mask, (img_orig.shape[1], img_orig.shape[0]))
    
    # 二值化 mask (255=Edit, 0=Keep)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 反转 mask 用于计算背景指标 (255=Keep, 0=Edit)
    background_mask = 255 - binary_mask
    
    # 2. 计算 Mask 区域指标 (Edit Intensity)
    # 仅在 mask 区域计算像素差异
    diff = cv2.absdiff(img_orig, img_edit)
    # mean in mask area
    mask_pixels = np.count_nonzero(binary_mask)
    if mask_pixels > 0:
        edit_intensity = np.sum(diff[binary_mask == 255]) / (mask_pixels * 3) # per pixel per channel
    else:
        edit_intensity = 0.0
        
    # 3. 计算 Background 指标 (Consistency)
    # Mask out edited area (make them black in both images)
    # 注意：SSIM 需要完整的矩形图像才有意义，这里简单的 mask out 可能有些不准，
    # 但作为对比是 OK 的。更严谨的做法是只计算 bounding box 或 masked SSIM。
    # 这里我们简单把编辑区域涂黑，然后算全图指标。
    
    img_orig_bg = cv2.bitwise_and(img_orig, img_orig, mask=background_mask)
    img_edit_bg = cv2.bitwise_and(img_edit, img_edit, mask=background_mask)
    
    # PSNR
    try:
        score_psnr = psnr(img_orig_bg, img_edit_bg, data_range=255)
    except Exception:
        score_psnr = 0.0
        
    # SSIM (multichannel)
    try:
        score_ssim = ssim(img_orig_bg, img_edit_bg, channel_axis=2, data_range=255)
    except Exception:
        score_ssim = 0.0
    
    return {
        "psnr": score_psnr,
        "ssim": score_ssim,
        "edit_intensity": edit_intensity
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SDXL Inpainting Session")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    parser.add_argument("--storage", type=str, default="storage/sessions", help="Storage directory")
    args = parser.parse_args()
    
    session_dir = Path(args.storage) / args.session
    if not session_dir.exists():
        print(f"Session {args.session} not found.")
        return
        
    output_file = session_dir / "evaluation.csv"
    
    # 读取版本号（兼容新旧存储：v{n}.png）
    version_re = re.compile(r"^v(\d+)\.png$")
    versions = []
    for p in session_dir.glob("v*.png"):
        m = version_re.match(p.name)
        if m:
            versions.append(int(m.group(1)))
    versions = sorted(set(versions))
    
    results = []
    
    # v0 是基础，不用评估
    # 从 v1 开始评估 (相对于 v0 或 v(i-1)?)
    # 通常评估是 "Original (Before Edit)" vs "Edited"
    # 这里我们对比 v(i) 和 v(i-1)
    
    print(f"Evaluating session {args.session}...")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['version', 'edit_text', 'psnr', 'ssim', 'edit_intensity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for curr_ver in versions[1:]:
            prev_ver = curr_ver - 1
            if prev_ver not in versions:
                continue
            
            curr_img_path = str(session_dir / f"v{curr_ver}.png")
            prev_img_path = str(session_dir / f"v{prev_ver}.png")
            mask_path = str(session_dir / f"v{curr_ver}_mask.png")
            
            if not Path(mask_path).exists():
                print(f"Skipping v{curr_ver}: No mask found (maybe generated?)")
                continue
                
            metrics = calculate_metrics(prev_img_path, curr_img_path, mask_path)
            
            edit_text = None
            # 新格式: v{n}_card.json (record 包含 edit_text 与 card)
            card_path = session_dir / f"v{curr_ver}_card.json"
            if card_path.exists():
                record = json.loads(card_path.read_text(encoding="utf-8"))
                edit_text = record.get("edit_text") or (record.get("card") or {}).get("edit_text")
            else:
                # 旧格式: v{n}_params.json（直接把 card 展平）
                params_path = session_dir / f"v{curr_ver}_params.json"
                if params_path.exists():
                    params = json.loads(params_path.read_text(encoding="utf-8"))
                    edit_text = params.get("edit_text")
            
            row = {
                "version": curr_ver,
                "edit_text": edit_text or f"v{curr_ver}",
                "psnr": f"{metrics['psnr']:.2f}",
                "ssim": f"{metrics['ssim']:.4f}",
                "edit_intensity": f"{metrics['edit_intensity']:.2f}"
            }
            results.append(row)
            writer.writerow(row)
            print(f"v{curr_ver}: PSNR={row['psnr']}, SSIM={row['ssim']}")
            
    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
