"""
Stage S0: Preprocess
Converts input image to grayscale, scale-normalizes to fixed long side (~2000px),
and performs dual binarization (dark for geometry, any_ink for thin grey lines).
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any


def preprocess_image(
    img_bgr: np.ndarray,
    target_long_side: int = 2000,
    dark_thresh: int = 110,
    ink_thresh: int = 210
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    S0 Preprocessing:
    - Scale normalizes image so max(h, w) == target_long_side.
    - Converts to grayscale.
    - Dual binarization:
        - `dark`: threshold ~110 for solid workpiece geometry.
        - `any_ink`: threshold ~210 for thin grey dimension & witness lines.
    
    Returns:
        gray: Grayscale normalized image (h, w)
        dark: Binary mask of solid geometry (h, w)
        any_ink: Binary mask of all strokes (geometry + annotations) (h, w)
        scale: Scale factor applied to original image
        orig_shape: (orig_h, orig_w)
    """
    orig_h, orig_w = img_bgr.shape[:2]
    max_dim = max(orig_h, orig_w)
    
    if max_dim != target_long_side and max_dim > 0:
        scale = float(target_long_side) / float(max_dim)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        img_scaled = cv2.resize(img_bgr, (new_w, new_h), interpolation=interp)
    else:
        scale = 1.0
        img_scaled = img_bgr.copy()
        
    gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
    
    # Dual binarization:
    # In standard drawings, background is light (>220) and ink is dark (<210).
    # dark: threshold ~100-110 for dark geometry
    # any_ink: threshold ~200-210 for any ink including light dimension lines
    ink_pixels = gray[gray < 235]
    if len(ink_pixels) > 50:
        p10 = float(np.percentile(ink_pixels, 10))
        p70 = float(np.percentile(ink_pixels, 70))
        if p10 > 80:
            actual_dark = int(min(185, max(dark_thresh, p70)))
        else:
            actual_dark = dark_thresh
    else:
        actual_dark = dark_thresh
    
    _, dark = cv2.threshold(gray, actual_dark, 255, cv2.THRESH_BINARY_INV)
    _, any_ink = cv2.threshold(gray, ink_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Boundary margin clearing (2px border)
    dark[:2, :] = 0; dark[-2:, :] = 0; dark[:, :2] = 0; dark[:, -2:] = 0
    any_ink[:2, :] = 0; any_ink[-2:, :] = 0; any_ink[:, :2] = 0; any_ink[:, -2:] = 0
    
    return gray, dark, any_ink, scale, (orig_h, orig_w)
