import numpy as np
from PIL import Image

from sdxl_app.engine.mask_utils import MaskProcessor


def test_mask_preprocess_alpha_square():
    # RGBA mask: transparent background, opaque square
    w, h = 64, 64
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[16:32, 20:40, 3] = 255
    mask = Image.fromarray(rgba, mode="RGBA")

    mp = MaskProcessor()
    processed = mp.preprocess(mask, grow_pixels=0, blur_sigma=0.0, invert=False)

    assert processed.mask_for_inpaint.mode == "L"
    assert processed.alpha.shape == (h, w)
    assert processed.alpha.dtype == np.float32

    # Inside square should be 255/1.0, outside 0
    assert processed.mask_for_inpaint.getpixel((25, 20)) == 255
    assert processed.mask_for_inpaint.getpixel((0, 0)) == 0
    assert processed.alpha[20, 25] == 1.0
    assert processed.alpha[0, 0] == 0.0


def test_alpha_blend_basic():
    w, h = 32, 32
    original = Image.new("RGB", (w, h), (0, 0, 255))
    edited = Image.new("RGB", (w, h), (255, 0, 0))

    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[8:16, 8:16] = 1.0

    mp = MaskProcessor()
    out = mp.alpha_blend(edited, original, alpha)
    assert out.size == (w, h)
    assert out.getpixel((10, 10)) == (255, 0, 0)
    assert out.getpixel((0, 0)) == (0, 0, 255)
