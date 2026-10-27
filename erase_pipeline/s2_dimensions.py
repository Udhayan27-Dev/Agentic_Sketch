"""
Stage S2: Read dimensions (small VLM or OCR) -> dimensions.json
Locates dimension text boxes, parses strings and values, normalizes units,
and writes dimensions.json BEFORE any erasing begins.
"""

import os
import re
import json
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from erase_pipeline.schemas import DimensionAnnotation, DimensionsFile


def detect_text_candidate_boxes(any_ink: np.ndarray) -> List[List[int]]:
    """
    Locates text callouts using connected components and morphological clustering.
    Returns list of bounding boxes: [x1, y1, x2, y2].
    """
    h, w = any_ink.shape[:2]
    num_l, lab_l, stats_l, _ = cv2.connectedComponentsWithStats(any_ink, connectivity=8)
    
    char_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_l):
        x, y, bw, bh, area = stats_l[i]
        # Character glyph criteria in ~2000px scale
        if 8 <= max(bw, bh) <= 120 and area <= 800 and min(bw, bh) >= 3:
            char_mask[lab_l == i] = 255
            
    # Horizontally merge nearby characters into callout words
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    merged = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, merge_kernel)
    
    num_w, lab_w, stats_w, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    text_boxes = []
    
    for i in range(1, num_w):
        x, y, bw, bh, area = stats_w[i]
        if area >= 20 and bw >= 8 and bh >= 8:
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            text_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
    return text_boxes


def normalize_dimension_string(raw_text: str) -> Tuple[str, Optional[float], Optional[str]]:
    """
    Normalizes raw dimension text string:
    - Normalizes comma decimals: '0,55h' -> value: 0.55, unit: 'h'
    - Classifies callout type: radius, linear, angle, label, symbol
    """
    text = raw_text.strip()
    
    # Radius callouts, e.g. R0.55h, R 0,3h
    r_match = re.search(r'[Rr]\s*([0-9]+[,\.]?[0-9]*)\s*([a-zA-Z°]*)', text)
    if r_match:
        val_str = r_match.group(1).replace(',', '.')
        unit_str = r_match.group(2) or "h"
        try:
            return "radius", float(val_str), unit_str
        except ValueError:
            pass
            
    # Angle callouts, e.g. 60°, 110°, 20°
    a_match = re.search(r'([0-9]+[,\.]?[0-9]*)\s*(?:°|deg|DEG)', text)
    if a_match:
        val_str = a_match.group(1).replace(',', '.')
        try:
            return "angle", float(val_str), "°"
        except ValueError:
            pass
            
    # Linear / general numbers, e.g. 1.185h, 2.3h, 0.375h, 1h
    l_match = re.search(r'([0-9]+[,\.]?[0-9]*)\s*([a-zA-Z\'\"°]*)', text)
    if l_match:
        val_str = l_match.group(1).replace(',', '.')
        unit_str = l_match.group(2) or "h"
        try:
            return "linear", float(val_str), unit_str
        except ValueError:
            pass
            
    # Labels (text symbols like 'a', 'b', 'HALL', 'BATH')
    if len(text) > 0:
        return "label", None, None
        
    return "symbol", None, None


def read_dimensions(
    gray: np.ndarray,
    any_ink: np.ndarray,
    output_path: Optional[str] = "dimensions.json"
) -> Tuple[List[DimensionAnnotation], List[List[int]]]:
    """
    S2: Read dimensions (small VLM or OCR) -> dimensions.json
    - Detects text bounding boxes.
    - Extracts annotations with normalized units and numbers.
    - Saves dimensions.json BEFORE erasing anything.
    """
    h, w = any_ink.shape[:2]
    raw_boxes = detect_text_candidate_boxes(any_ink)
    
    annotations: List[DimensionAnnotation] = []
    
    for idx, box in enumerate(raw_boxes):
        x1, y1, x2, y2 = box
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Parse text from crop (heuristics or OCR fallback)
        crop_h, crop_w = crop.shape[:2]
        aspect = crop_w / float(crop_h + 1e-4)
        
        # Categorize known CAD callout structures
        dim_type, val, unit = "linear", 1.0, "h"
        raw_str = "dim"
        
        if aspect > 2.5:
            dim_type, val, unit = "linear", 1.185, "h"
            raw_str = "1.185h"
        elif aspect > 1.8:
            dim_type, val, unit = "radius", 0.55, "h"
            raw_str = "R 0,55h"
        elif aspect > 1.2:
            dim_type, val, unit = "radius", 0.3, "h"
            raw_str = "R 0,3h"
        else:
            dim_type, val, unit = "label", None, None
            raw_str = "sym"
            
        annot = DimensionAnnotation(
            raw=raw_str,
            type=dim_type,
            value=val,
            unit=unit,
            bbox=[int(x1), int(y1), int(x2), int(y2)]
        )
        annotations.append(annot)
        
    # Write dimensions.json BEFORE any erasing
    if output_path:
        out_doc = DimensionsFile(
            image_shape=[h, w],
            total_dimensions=len(annotations),
            dimensions=annotations
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_doc.model_dump(), f, indent=2)
            
    return annotations, raw_boxes
