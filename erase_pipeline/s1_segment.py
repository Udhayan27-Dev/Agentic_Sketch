"""
Stage S1: Segment into labeled pieces
Computes skeleton, breaks graph at degree >= 3 junctions into edges,
and extracts topological features for each segment.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import Dict, List, Tuple, Any
from erase_pipeline.schemas import SegmentFeatures


def segment_ink(
    any_ink: np.ndarray,
    gray: np.ndarray,
    dark: np.ndarray,
    text_boxes: List[List[int]] = None
) -> Tuple[Dict[int, SegmentFeatures], np.ndarray, np.ndarray, np.ndarray]:
    """
    S1 Segmentation:
    - Computes topological skeleton and distance transform.
    - Locates degree >= 3 junction nodes and splits graph into individual edges.
    - Extracts stroke_width, mean_gray, fill_ratio, loop status, endpoints for each edge.
    
    Returns:
        segments: Dict mapping segment_id -> SegmentFeatures
        edge_labels: 2D int32 map of segment IDs along skeleton/ink
        skel: Full skeleton binary mask
        junctions: Binary mask of degree >= 3 junction nodes
    """
    h, w = any_ink.shape[:2]
    text_boxes = text_boxes or []
    
    # 1. Morphological Skeleton
    skel = skeletonize(any_ink > 0).astype(np.uint8)
    
    # Distance transform for stroke width estimation
    dist = cv2.distanceTransform(any_ink, cv2.DIST_L2, 5)
    
    # 2. Graph degree analysis via 3x3 neighbor convolution
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel, -1, kernel)
    
    # Degree >= 3 nodes (junctions) and degree 1 nodes (endpoints)
    junctions = ((skel > 0) & (filtered >= 13)).astype(np.uint8) * 255
    endpoints = ((skel > 0) & (filtered == 11)).astype(np.uint8) * 255
    
    # 3. Closed loops detection in dark geometry
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    closed_loop_mask = np.zeros((h, w), dtype=np.uint8)
    if hierarchy is not None:
        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(closed_loop_mask, [cnt], -1, 255, 3)
                
    # 4. Break skeleton graph at degree >= 3 junctions into isolated edges
    # Dilate junctions slightly (3x3) to cleanly disconnect edges
    junctions_dil = cv2.dilate(junctions, np.ones((3, 3), np.uint8))
    edges_skel = cv2.subtract(skel, (junctions_dil > 0).astype(np.uint8))
    
    num_edges, edge_labels, stats, _ = cv2.connectedComponentsWithStats(edges_skel, connectivity=8)
    
    segments: Dict[int, SegmentFeatures] = {}
    
    for seg_id in range(1, num_edges):
        x, y, bw, bh, length = stats[seg_id]
        if length < 2:
            continue
            
        seg_mask = (edge_labels == seg_id)
        
        # Stroke width: median of diameter (2 * dist) along skeleton
        widths = 2.0 * dist[seg_mask]
        stroke_w = float(np.median(widths)) if len(widths) > 0 else 1.0
        
        # Mean gray intensity
        mean_g = float(np.mean(gray[seg_mask]))
        
        # Fill ratio: area / (w * h)
        bbox_area = max(1, bw * bh)
        fill_r = float(length) / float(bbox_area)
        
        # Closed loop check: does edge overlap with closed loop contours?
        loop_overlap = np.count_nonzero(closed_loop_mask[seg_mask])
        is_loop = (loop_overlap > 0.5 * length)
        
        # Free endpoints: check if edge endpoints touch global endpoints
        seg_endpoints_count = np.count_nonzero(endpoints[seg_mask])
        
        # Text box check: does bounding box touch any OCR text box?
        touches_text = False
        pad = 8
        seg_x1, seg_y1, seg_x2, seg_y2 = x - pad, y - pad, x + bw + pad, y + bh + pad
        for tb in text_boxes:
            tx1, ty1, tx2, ty2 = tb
            # Bounding box intersection check
            if not (seg_x2 < tx1 or seg_x1 > tx2 or seg_y2 < ty1 or seg_y1 > ty2):
                touches_text = True
                break
                
        feat = SegmentFeatures(
            id=seg_id,
            area=int(length),
            bbox=[int(x), int(y), int(bw), int(bh)],
            stroke_width=round(stroke_w, 2),
            mean_gray=round(mean_g, 2),
            fill_ratio=round(fill_r, 4),
            is_on_closed_loop=bool(is_loop),
            touches_text_box=bool(touches_text),
            free_endpoints=int(seg_endpoints_count)
        )
        segments[seg_id] = feat
        
    return segments, edge_labels, skel, junctions
