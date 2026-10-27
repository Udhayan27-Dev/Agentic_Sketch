"""
CAD Inspector Agent Controller
Orchestrates the 7-Stage Dimension-Erase Pipeline (Mask-only, no redrawing):
S0: Preprocess (gray, dual binarize: dark vs any_ink)
S1: Segment into labeled pieces (skel, break skeleton at degree >= 3)
S2: Read dimensions -> dimensions.json (saved first)
S3: Classify segments (rules first, VLM for unsure)
S4: Build masks (erase = render(erase_edges) | dilate(text_boxes, 3), protect = dilate(thick_dark & on_closed_loop, 1), final = erase & ~protect)
S5: Apply (clean = original.copy(); clean[final > 0] = 255; original geometry pixels untouched)
S6: Verify and fix (repeat once, write report.json with leftovers)
"""

import cv2
import numpy as np
from typing import Optional
from erase_pipeline.pipeline import DimensionErasePipeline


class CADInspectorAgent:
    def __init__(self, agent_name: str = "CAD-Geometry-Inspector-V1", target_long_side: int = 2000):
        self.agent_name = agent_name
        self.pipeline = DimensionErasePipeline(target_long_side=target_long_side)

    def inspect_and_isolate(
        self,
        image_bytes: bytes,
        ground_truth_path: Optional[str] = None,
        save_artifacts_dir: Optional[str] = None
    ) -> dict:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode uploaded image file")

        result = self.pipeline.run(
            img_bgr,
            ground_truth_path=ground_truth_path,
            save_artifacts_dir=save_artifacts_dir
        )
        
        # Format clean JSON response for API endpoints
        return {
            "session_id": result["session_id"],
            "agent": self.agent_name,
            "total_execution_ms": result["total_execution_ms"],
            "inspection_history": result["inspection_history"],
            "metrics": result["metrics"],
            "isolated_geometry_base64": result["isolated_geometry_base64"],
            "clean_bgr_base64": result["clean_bgr_base64"],
            "dimensions": result["dimensions"],
            "report": result["report"]
        }
