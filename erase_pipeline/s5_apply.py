"""
Stage S5: Apply (Mask-only, NO redrawing)
Whites out annotations using the final mask. Original geometry pixels remain untouched.
clean = original.copy()
clean[final > 0] = 255
"""

import cv2
import numpy as np
from typing import Tuple


def apply_mask(
    original_bgr: np.ndarray,
    final_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    S5: Apply
    - Modifies original image by whitening out final mask pixels (clean[final > 0] = 255).
    - Preserves 100% of original geometry pixel values without any artificial redrawing.
    - Also produces clean_rgba on transparent 0-color background for CAD drafting viewer.
    
    Returns:
        clean_bgr: Original image with dimensions whitened out (255, 255, 255).
        clean_rgba: Transparent RGBA image where whitened background is alpha=0.
    """
    h, w = original_bgr.shape[:2]
    
    # 1. White-out only: original geometry pixels untouched!
    clean_bgr = original_bgr.copy()
    clean_bgr[final_mask > 0] = [255, 255, 255]
    
    # 2. Transparent CAD Canvas representation:
    # Any pixel not whitened out (and darker than background) is preserved
    clean_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    is_geom = (gray < 220)
    
    clean_rgba[is_geom, :3] = original_bgr[is_geom]
    clean_rgba[is_geom, 3] = 255
    
    return clean_bgr, clean_rgba
