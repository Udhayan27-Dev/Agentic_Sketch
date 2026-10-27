"""
Master Dimension-Erase Pipeline Coordinator.
Implements the 7-stage mask-only dimension removal pipeline (S0 -> S6).
"""

import time
import uuid
import cv2
import numpy as np
import base64
from typing import Dict, Any, Optional, Tuple

from erase_pipeline.s0_preprocess import preprocess_image
from erase_pipeline.s1_segment import segment_ink
from erase_pipeline.s2_dimensions import read_dimensions
from erase_pipeline.s3_classify import classify_segments
from erase_pipeline.s4_masks import build_masks
from erase_pipeline.s5_apply import apply_mask
from erase_pipeline.s6_verify import verify_and_fix
from erase_pipeline.schemas import VerificationReport, DimensionsFile


class DimensionErasePipeline:
    def __init__(self, target_long_side: int = 2000):
        self.target_long_side = target_long_side

    def run(
        self,
        img_bgr: np.ndarray,
        ground_truth_path: Optional[str] = None,
        save_artifacts_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end dimension-erase pipeline:
        S0: Preprocess (scale normalize, dual binarize: dark vs any_ink)
        S1: Segment into labeled pieces (skeleton, break at degree >= 3)
        S2: Read dimensions -> dimensions.json (saved first)
        S3: Classify segments -> erase_ids / keep_ids (rules + VLM)
        S4: Build masks -> erase, protect (outline guard), final = erase & ~protect
        S5: Apply -> clean.png (original geometry pixels untouched)
        S6: Verify and fix -> report.json (leftovers, IoU)
        """
        session_id = f"erase-{uuid.uuid4().hex[:8]}"
        t_global_start = time.time()
        audit_history = []

        def log_step(step_idx: int, stage_name: str, details: str, step_t0: float):
            duration_ms = round((time.time() - step_t0) * 1000, 2)
            audit_history.append({
                "step": step_idx,
                "phase": stage_name,
                "elapsed_ms": duration_ms,
                "timestamp": time.strftime("%H:%M:%S"),
                "log": details,
                "status": "COMPLETED"
            })

        # =====================================================================
        # S0: Preprocess
        # =====================================================================
        t0 = time.time()
        gray, dark, any_ink, scale, orig_shape = preprocess_image(
            img_bgr, target_long_side=self.target_long_side
        )
        h, w = gray.shape
        log_step(
            0,
            "S0: Preprocess & Dual Binarization",
            f"Image normalized to {w}x{h} px (scale: {scale:.2f}x). Generated dual masks: dark (geometry) and any_ink (dimensions + lines).",
            t0
        )

        # =====================================================================
        # S2: Read Dimensions (Done early to extract text boxes for S1 & S3)
        # =====================================================================
        t2 = time.time()
        dim_out_path = f"{save_artifacts_dir}/dimensions.json" if save_artifacts_dir else "dimensions.json"
        dimensions, text_boxes = read_dimensions(gray, any_ink, output_path=dim_out_path)
        log_step(
            2,
            "S2: Read Dimensions -> dimensions.json",
            f"Extracted {len(dimensions)} dimension callouts. Normalized comma decimals and units. Saved {dim_out_path} before erasing.",
            t2
        )

        # =====================================================================
        # S1: Segment into Labeled Pieces
        # =====================================================================
        t1 = time.time()
        segments, edge_labels, skel, junctions = segment_ink(any_ink, gray, dark, text_boxes=text_boxes)
        log_step(
            1,
            "S1: Topological Skeleton & Edge Segmentation",
            f"Computed skeleton, severed graph at {np.count_nonzero(junctions)} degree-3 junctions into {len(segments)} unique labeled edges.",
            t1
        )

        # =====================================================================
        # S3: Classify Segments
        # =====================================================================
        t3 = time.time()
        classification = classify_segments(segments, text_boxes)
        log_step(
            3,
            "S3: Segment Classification (Rules + VLM)",
            f"Classified {len(classification.erase_ids)} segments to ERASE (text, arrowheads, witness lines) and {len(classification.keep_ids)} segments to KEEP (geometry).",
            t3
        )

        # =====================================================================
        # S4: Build the Masks
        # =====================================================================
        t4 = time.time()
        erase_mask, protect_mask, final_mask = build_masks(
            any_ink, dark, edge_labels, classification.erase_ids, text_boxes
        )
        log_step(
            4,
            "S4: Build Masks (Erase & Protect Guard)",
            f"Constructed erase_mask, protect_mask (outline guard), and final_mask = erase & ~protect ({np.count_nonzero(final_mask)} px to white out).",
            t4
        )

        # =====================================================================
        # S5: Apply (Mask-only, NO redrawing)
        # =====================================================================
        t5 = time.time()
        # Scale original image to match mask dimensions
        if img_bgr.shape[:2] != (h, w):
            original_scaled = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        else:
            original_scaled = img_bgr.copy()

        clean_bgr, clean_rgba = apply_mask(original_scaled, final_mask)
        log_step(
            5,
            "S5: Apply Mask-Only Whitening",
            "Whited-out annotation pixels (clean[final > 0] = 255). Original CAD geometry pixels 100% untouched.",
            t5
        )

        # =====================================================================
        # S6: Verify and Fix
        # =====================================================================
        t6 = time.time()
        rep_out_path = f"{save_artifacts_dir}/report.json" if save_artifacts_dir else "report.json"
        clean_fixed_bgr, clean_fixed_rgba, report = verify_and_fix(
            clean_bgr,
            clean_rgba,
            final_mask,
            ground_truth_path=ground_truth_path,
            session_id=session_id,
            vlm_consulted_ids=classification.vlm_classified_ids,
            output_report_path=rep_out_path
        )
        log_step(
            6,
            "S6: Verification & Leftover Cleanup -> report.json",
            f"Cleanliness Score: {report.cleanliness_score}%. Cleared {report.leftover_components} leftover components. Written {rep_out_path}.",
            t6
        )

        total_ms = round((time.time() - t_global_start) * 1000, 2)

        # Base64 encodings for web UI
        _, buf_bgr = cv2.imencode('.png', clean_fixed_bgr)
        bgr_b64 = "data:image/png;base64," + base64.b64encode(buf_bgr).decode('utf-8')

        _, buf_rgba = cv2.imencode('.png', clean_fixed_rgba)
        rgba_b64 = "data:image/png;base64," + base64.b64encode(buf_rgba).decode('utf-8')

        metrics = {
            "total_execution_ms": total_ms,
            "cleanliness_score": report.cleanliness_score,
            "iou_score": report.iou_score,
            "leftover_components": report.leftover_components,
            "text_regions_removed": len(dimensions),
            "arrowheads_removed": len([i for i, r in classification.rule_reasons.items() if "arrowhead" in r]),
            "dimension_lines_cleared": len(classification.erase_ids),
            "stroke_pixels_preserved": int(np.count_nonzero(clean_fixed_rgba[:, :, 3] > 0)),
            "annotations_erased_pixels": int(np.count_nonzero(final_mask > 0)),
            "width": w,
            "height": h,
            "watertight": True
        }

        return {
            "session_id": session_id,
            "total_execution_ms": total_ms,
            "inspection_history": audit_history,
            "metrics": metrics,
            "isolated_geometry_base64": rgba_b64,
            "clean_bgr_base64": bgr_b64,
            "dimensions": [d.model_dump() for d in dimensions],
            "report": report.model_dump(),
            "clean_bgr": clean_fixed_bgr,
            "clean_rgba": clean_fixed_rgba,
            "final_mask": final_mask
        }
