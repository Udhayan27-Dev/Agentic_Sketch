"""
Stage S4: Build the masks
Constructs erase_mask, protect_mask (outline guard), and final_mask.
final = erase & ~protect
"""

import cv2
import numpy as np
from typing import List, Set, Tuple


def build_masks(
    any_ink: np.ndarray,
    dark: np.ndarray,
    edge_labels: np.ndarray,
    erase_ids: List[int],
    text_boxes: List[List[int]],
    on_closed_loop_mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    S4 Mask Construction:
    - erase   = render(erase_edges) | dilate(text_boxes, 3) | arrowheads
    - protect = dilate(thick_dark & on_closed_loop, 1)      # outline guard
    - final   = erase & ~protect
    
    Returns:
        erase_mask: Binary mask of all candidate annotations to remove
        protect_mask: Guard mask of structural workpiece outline
        final_mask: Final pixel mask to be whitened out (guarded from part outline)
    """
    h, w = any_ink.shape[:2]
    
    # 1. Render erase edges
    erase_set = set(erase_ids)
    erase_edges = np.isin(edge_labels, list(erase_set)).astype(np.uint8) * 255
    # Expand slightly by 3px to cover full stroke width of thin lines
    erase_edges_dil = cv2.dilate(erase_edges, np.ones((3, 3), np.uint8))
    
    # 2. Render text boxes dilated by 3-5px
    text_boxes_mask = np.zeros((h, w), dtype=np.uint8)
    for box in text_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(text_boxes_mask, (x1, y1), (x2, y2), 255, -1)
    text_boxes_dil = cv2.dilate(text_boxes_mask, np.ones((5, 5), np.uint8))
    
    # 3. Detect arrowheads directly from ink contours
    arrow_mask = np.zeros((h, w), dtype=np.uint8)
    cnts, _ = cv2.findContours(any_ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if 8 <= area <= 300:
            hull = cv2.convexHull(c)
            solidity = area / (cv2.contourArea(hull) + 1e-4)
            x, y, bw, bh = cv2.boundingRect(c)
            ar = max(bw, bh) / (min(bw, bh) + 1e-4)
            if solidity > 0.45 and 1.2 <= ar <= 4.5:
                cv2.drawContours(arrow_mask, [c], -1, 255, -1)
    arrow_mask_dil = cv2.dilate(arrow_mask, np.ones((3, 3), np.uint8))
    
    # Combine candidate erasures
    erase = cv2.bitwise_or(erase_edges_dil, cv2.bitwise_or(text_boxes_dil, arrow_mask_dil))
    # Mask erase to only where ink actually exists
    erase = cv2.bitwise_and(erase, any_ink)
    
    # 4. Build protect mask (outline guard)
    # thick_dark & on_closed_loop
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    # Thick strokes belong to the workpiece outline
    max_d = np.max(dist) if np.max(dist) > 0 else 3.0
    thick_thresh = max(1.8, 0.45 * max_d)
    thick_geom = (dist >= thick_thresh).astype(np.uint8) * 255
    
    # Outline guard: dilate thick geometry by 2px
    protect = cv2.dilate(thick_geom, np.ones((3, 3), np.uint8), iterations=1)
    
    # 5. Final = erase & ~protect
    # This prevents touching arrowheads and leader lines from creating holes in outline!
    final = cv2.bitwise_and(erase, cv2.bitwise_not(protect))
    
    return erase, protect, final
