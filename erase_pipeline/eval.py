"""
Evaluation script: IoU and Chamfer distance across a dataset of (input, ground_truth) pairs.
"""

import os
import glob
import json
import cv2
import numpy as np
from erase_pipeline.pipeline import DimensionErasePipeline
from erase_pipeline.s6_verify import compute_iou, compute_chamfer


def evaluate_folder(input_dir: str, ground_truth_dir: str, output_report: str = "eval_results.json"):
    pipeline = DimensionErasePipeline(target_long_side=1500)
    results = []

    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    input_files = []
    for ext in exts:
        input_files.extend(glob.glob(os.path.join(input_dir, ext)))

    for in_path in sorted(input_files):
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        # Look for matching ground truth file
        gt_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            cand = os.path.join(ground_truth_dir, base_name + ext)
            if os.path.exists(cand):
                gt_path = cand
                break

        img = cv2.imread(in_path)
        if img is None:
            continue

        res = pipeline.run(img, ground_truth_path=gt_path)
        results.append({
            "image": os.path.basename(in_path),
            "has_gt": gt_path is not None,
            "iou": res["metrics"]["iou_score"],
            "leftovers": res["metrics"]["leftover_components"],
            "cleanliness": res["metrics"]["cleanliness_score"],
            "latency_ms": res["metrics"]["total_execution_ms"]
        })

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluated {len(results)} images. Results saved to {output_report}")
    return results


if __name__ == "__main__":
    import sys
    in_d = sys.argv[1] if len(sys.argv) > 1 else "."
    gt_d = sys.argv[2] if len(sys.argv) > 2 else "."
    evaluate_folder(in_d, gt_d)
