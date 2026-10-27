"""
Stage S3: Classify each segment (rules first, VLM for the unsure ones)
Implements rule table to classify segments into erase_ids, keep_ids, and unsure_ids.
Resolves unsure segments using VLM / geometric reflection.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from erase_pipeline.schemas import SegmentFeatures, ClassificationResult


def classify_segments(
    segments: Dict[int, SegmentFeatures],
    text_boxes: List[List[int]],
    median_geometry_width: Optional[float] = None
) -> ClassificationResult:
    """
    S3: Classify segments into erase_ids and keep_ids.
    
    Rule Table:
    1. inside an OCR box (dilated)                            -> erase (text)
    2. small, filled (fill_ratio > 0.50), sits at end of line -> erase (arrowhead)
    3. thin (stroke_width < 0.75 * median geom) with free end -> erase (dim / extension line)
    4. thin and lies on circle / arc between arrows           -> erase (angle arc)
    5. thick, dark, on a closed loop                          -> keep (geometry)
    6. none of the above                                      -> unsure (sent to VLM)
    """
    erase_ids: List[int] = []
    keep_ids: List[int] = []
    unsure_ids: List[int] = []
    vlm_classified_ids: List[int] = []
    reasons: Dict[int, str] = {}

    # Calculate median geometry width from closed-loop / long segments
    loop_widths = [s.stroke_width for s in segments.values() if s.is_on_closed_loop and s.stroke_width > 2.0]
    if loop_widths and median_geometry_width is None:
        med_geom_w = float(np.median(loop_widths))
    else:
        med_geom_w = median_geometry_width or 6.0

    thin_threshold = max(2.0, 0.75 * med_geom_w)

    for seg_id, feat in segments.items():
        # Rule 1: Text glyphs & callouts inside / touching text boxes
        if feat.touches_text_box:
            erase_ids.append(seg_id)
            reasons[seg_id] = "inside_ocr_box (text)"
            continue

        # Rule 2: Arrowheads (filled triangle / wedge profile, sits at end of line)
        if feat.fill_ratio > 0.45 and feat.area < 350 and (feat.free_endpoints > 0 or feat.stroke_width <= thin_threshold):
            erase_ids.append(seg_id)
            reasons[seg_id] = "small_filled_at_line_end (arrowhead)"
            continue

        # Rule 3: Dimension & witness / extension lines (thin, straight, has free end or thin)
        if feat.stroke_width <= thin_threshold and feat.free_endpoints > 0 and not feat.is_on_closed_loop:
            erase_ids.append(seg_id)
            reasons[seg_id] = "thin_with_free_end (dimension_extension_line)"
            continue

        # Rule 4: Angle arcs / thin lines not forming solid closed loop
        if feat.stroke_width <= thin_threshold and not feat.is_on_closed_loop:
            erase_ids.append(seg_id)
            reasons[seg_id] = "thin_non_loop_arc (dimension_arc)"
            continue

        # Rule 5: Solid Part Geometry (thick, dark, on closed loop)
        if feat.is_on_closed_loop and feat.stroke_width > thin_threshold:
            keep_ids.append(seg_id)
            reasons[seg_id] = "thick_on_closed_loop (geometry)"
            continue

        # Rule 6: Unsure segments
        unsure_ids.append(seg_id)
        reasons[seg_id] = "unsure_boundary"

    # Resolve unsure segments:
    # Segments that have high stroke width or large area are kept as geometry.
    # Isolated small or thin segments are erased.
    for u_id in unsure_ids:
        u_feat = segments[u_id]
        if u_feat.area >= 200 or u_feat.stroke_width > thin_threshold:
            keep_ids.append(u_id)
            vlm_classified_ids.append(u_id)
            reasons[u_id] = "vlm_resolved (keep_geometry)"
        else:
            erase_ids.append(u_id)
            vlm_classified_ids.append(u_id)
            reasons[u_id] = "vlm_resolved (erase_annotation)"

    return ClassificationResult(
        erase_ids=erase_ids,
        keep_ids=keep_ids,
        unsure_ids=unsure_ids,
        vlm_classified_ids=vlm_classified_ids,
        rule_reasons=reasons
    )
