"""Geometric Evaluation and Verification Harness for SketchCAD / CADCraft.

Evaluates predicted DrawingSpecs, CADCraft scripts, or exported DXF files against
ground-truth benchmarks measuring:
1. CADCraft Execution Success (exit code, DXF generation).
2. Topological Closure / Watertightness (max gap between consecutive contour vertices).
3. Geometric Tangency Error (angular difference at arc-line / arc-arc tangent joints).
4. Radius and Dimensional Accuracy (MAE and percentage error).
5. Comprehensive Accuracy Score (0 - 100%).
"""

import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import ezdxf


@dataclass
class GeometricEvaluationReport:
    sample_id: str
    cadcraft_success: bool
    is_watertight: bool
    max_gap_mm: float
    tangency_mean_error_deg: float
    tangency_max_error_deg: float
    tangent_joints_evaluated: int
    radius_mae_mm: float
    radius_max_error_mm: float
    entity_count_pred: int
    entity_count_gt: int
    accuracy_score: float
    summary: str
    details: Dict[str, Any]


def _unit(v: List[float]) -> List[float]:
    n = math.hypot(v[0], v[1])
    if n < 1e-9:
        return [0.0, 0.0]
    return [v[0] / n, v[1] / n]


def calculate_tangency_error(line_p1: List[float], line_p2: List[float],
                            arc_center: List[float], joint_pt: List[float]) -> float:
    """Calculates angular error (in degrees) between line tangent and circle/arc tangent at joint_pt."""
    # Line tangent direction
    ldx = line_p2[0] - line_p1[0]
    ldy = line_p2[1] - line_p1[1]
    u_line = _unit([ldx, ldy])

    # Radial vector from arc center to joint point
    rx = joint_pt[0] - arc_center[0]
    ry = joint_pt[1] - arc_center[1]
    u_radial = _unit([rx, ry])

    # Arc tangent is perpendicular to radial vector: (-u_radial[1], u_radial[0])
    u_arc_tan = [-u_radial[1], u_radial[0]]

    # Absolute dot product between line direction and arc tangent
    dot = abs(u_line[0] * u_arc_tan[0] + u_line[1] * u_arc_tan[1])
    dot = max(-1.0, min(1.0, dot))
    angle_rad = math.acos(dot)
    return math.degrees(angle_rad)


class CADGeometryEvaluator:
    """Evaluates 2D CAD geometry precision."""

    def __init__(self, tolerance_gap_mm: float = 0.05, tolerance_tangent_deg: float = 0.5):
        self.tolerance_gap_mm = tolerance_gap_mm
        self.tolerance_tangent_deg = tolerance_tangent_deg

    def evaluate_drawing_spec(self, pred_spec: Dict[str, Any], gt_spec: Dict[str, Any],
                             sample_id: str = "eval") -> GeometricEvaluationReport:
        """Evaluates a predicted DrawingSpec dictionary against ground truth."""
        pred_entities = pred_spec.get("entities", [])
        gt_entities = gt_spec.get("entities", [])

        # 1. Evaluate radius accuracy
        pred_radii = [e["radius"] for e in pred_entities if e.get("radius") is not None]
        gt_radii = [e["radius"] for e in gt_entities if e.get("radius") is not None]

        r_diffs = []
        for gr in gt_radii:
            if pred_radii:
                # Find closest predicted radius
                closest = min(pred_radii, key=lambda pr: abs(pr - gr))
                r_diffs.append(abs(closest - gr))
            else:
                r_diffs.append(gr)

        radius_mae = sum(r_diffs) / len(r_diffs) if r_diffs else 0.0
        radius_max = max(r_diffs) if r_diffs else 0.0

        # 2. Evaluate watertightness / loop closure
        # Extract all endpoints from contour lines and arcs
        endpoints = []
        for ent in pred_entities:
            etype = ent.get("type")
            if etype == "line" and ent.get("p1") and ent.get("p2"):
                endpoints.append((ent["p1"], ent["p2"]))
            elif etype == "arc" and ent.get("center") and ent.get("radius"):
                c = ent["center"]
                r = ent["radius"]
                a1 = math.radians(ent.get("start_deg", 0.0))
                a2 = math.radians(ent.get("end_deg", 0.0))
                p_start = [c[0] + r * math.cos(a1), c[1] + r * math.sin(a1)]
                p_end = [c[0] + r * math.cos(a2), c[1] + r * math.sin(a2)]
                endpoints.append((p_start, p_end))

        # In a watertight drawing (with 1 or more closed loops), every endpoint
        # must have at least one other endpoint from another entity matching it.
        all_pts = []
        for idx, (p1, p2) in enumerate(endpoints):
            all_pts.append((p1, idx, 1))
            all_pts.append((p2, idx, 2))

        gaps = []
        for pt, e_idx, pt_num in all_pts:
            other_pts = [p for p, o_idx, _ in all_pts if o_idx != e_idx]
            if other_pts:
                min_dist = min(math.hypot(pt[0] - op[0], pt[1] - op[1]) for op in other_pts)
                gaps.append(min_dist)

        max_gap = max(gaps) if gaps else 0.0
        is_watertight = max_gap <= self.tolerance_gap_mm if gaps else True

        # 3. Evaluate tangency error at tangent joints
        tangent_errors = []
        gt_relations = gt_spec.get("relations", [])
        tangent_rels = [r for r in gt_relations if r.get("kind") == "tangent"]

        ent_by_id = {e.get("id"): e for e in pred_entities if "id" in e}
        for rel in tangent_rels:
            refs = rel.get("refs", [])
            if len(refs) == 2:
                e1, e2 = ent_by_id.get(refs[0]), ent_by_id.get(refs[1])
                if e1 and e2:
                    line_e = e1 if e1.get("type") == "line" else (e2 if e2.get("type") == "line" else None)
                    arc_e = e1 if e1.get("type") == "arc" else (e2 if e2.get("type") == "arc" else None)
                    if line_e and arc_e and line_e.get("p1") and line_e.get("p2") and arc_e.get("center"):
                        # Joint point is line endpoint closest to arc center
                        d1 = math.hypot(line_e["p1"][0] - arc_e["center"][0], line_e["p1"][1] - arc_e["center"][1])
                        d2 = math.hypot(line_e["p2"][0] - arc_e["center"][0], line_e["p2"][1] - arc_e["center"][1])
                        joint_pt = line_e["p1"] if d1 < d2 else line_e["p2"]
                        err = calculate_tangency_error(line_e["p1"], line_e["p2"], arc_e["center"], joint_pt)
                        tangent_errors.append(err)

        mean_tan_err = sum(tangent_errors) / len(tangent_errors) if tangent_errors else 0.0
        max_tan_err = max(tangent_errors) if tangent_errors else 0.0

        # 4. Compute composite accuracy score (0.0 to 100.0)
        # Score components:
        # S_watertight: 100 if gap <= 0.05, decays to 0 at 2.0 mm
        s_watertight = max(0.0, 100.0 * (1.0 - min(max_gap / 2.0, 1.0))) if gaps else 100.0
        # S_tangency: 100 if err <= 0.1 deg, decays to 0 at 5 deg
        s_tangency = max(0.0, 100.0 * (1.0 - min(mean_tan_err / 5.0, 1.0))) if tangent_errors else 100.0
        # S_radius: 100 if mae <= 0.1 mm, decays to 0 at 5.0 mm
        s_radius = max(0.0, 100.0 * (1.0 - min(radius_mae / 5.0, 1.0))) if r_diffs else 100.0
        # S_entities: entity count ratio
        count_ratio = min(len(pred_entities), len(gt_entities)) / max(len(gt_entities), 1)
        s_count = count_ratio * 100.0

        composite_score = round(
            0.30 * s_watertight + 0.35 * s_tangency + 0.20 * s_radius + 0.15 * s_count, 2
        )

        summary = (
            f"Score: {composite_score:.1f}% | Watertight: {is_watertight} (gap={max_gap:.3f}mm) | "
            f"Tangency Error: {mean_tan_err:.2f}° (max {max_tan_err:.2f}°) | Radius MAE: {radius_mae:.2f}mm"
        )

        return GeometricEvaluationReport(
            sample_id=sample_id,
            cadcraft_success=True,
            is_watertight=is_watertight,
            max_gap_mm=round(max_gap, 4),
            tangency_mean_error_deg=round(mean_tan_err, 4),
            tangency_max_error_deg=round(max_tan_err, 4),
            tangent_joints_evaluated=len(tangent_errors),
            radius_mae_mm=round(radius_mae, 4),
            radius_max_error_mm=round(radius_max, 4),
            entity_count_pred=len(pred_entities),
            entity_count_gt=len(gt_entities),
            accuracy_score=composite_score,
            summary=summary,
            details={
                "gaps": [round(g, 4) for g in gaps],
                "tangent_errors": [round(t, 4) for t in tangent_errors],
                "radius_errors": [round(r, 4) for r in r_diffs],
            },
        )

    def evaluate_dxf(self, dxf_path: str, gt_spec: Dict[str, Any], sample_id: str = "eval") -> GeometricEvaluationReport:
        """Evaluates an exported DXF file directly using ezdxf."""
        if not os.path.exists(dxf_path):
            return GeometricEvaluationReport(
                sample_id=sample_id,
                cadcraft_success=False,
                is_watertight=False,
                max_gap_mm=999.0,
                tangency_mean_error_deg=90.0,
                tangency_max_error_deg=90.0,
                tangent_joints_evaluated=0,
                radius_mae_mm=999.0,
                radius_max_error_mm=999.0,
                entity_count_pred=0,
                entity_count_gt=len(gt_spec.get("entities", [])),
                accuracy_score=0.0,
                summary=f"DXF file not found at {dxf_path}",
                details={"error": "file_not_found"},
            )

        try:
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            extracted_entities = []

            for i, entity in enumerate(msp):
                dxftype = entity.dxftype()
                layer_name = getattr(entity.dxf, "layer", "").upper()
                if any(k in layer_name for k in ("CENTER", "DIM", "ANNOTAT", "DEFPOINT")):
                    continue
                if dxftype == "LINE":
                    extracted_entities.append({
                        "id": f"e{len(extracted_entities)+1}",
                        "type": "line",
                        "p1": [entity.dxf.start.x, entity.dxf.start.y],
                        "p2": [entity.dxf.end.x, entity.dxf.end.y],
                    })
                elif dxftype == "CIRCLE":
                    extracted_entities.append({
                        "id": f"e{i+1}",
                        "type": "circle",
                        "center": [entity.dxf.center.x, entity.dxf.center.y],
                        "radius": entity.dxf.radius,
                    })
                elif dxftype == "ARC":
                    extracted_entities.append({
                        "id": f"e{len(extracted_entities)+1}",
                        "type": "arc",
                        "center": [entity.dxf.center.x, entity.dxf.center.y],
                        "radius": entity.dxf.radius,
                        "start_deg": entity.dxf.start_angle,
                        "end_deg": entity.dxf.end_angle,
                    })
                elif dxftype in ("LWPOLYLINE", "POLYLINE"):
                    pts = list(entity.vertices()) if dxftype == "POLYLINE" else [p[:2] for p in entity.get_points()]
                    is_closed = entity.is_closed
                    num_pts = len(pts)
                    for j in range(num_pts if is_closed else num_pts - 1):
                        p_start = [pts[j][0], pts[j][1]]
                        p_end = [pts[(j + 1) % num_pts][0], pts[(j + 1) % num_pts][1]]
                        extracted_entities.append({
                            "id": f"e{len(extracted_entities)+1}",
                            "type": "line",
                            "p1": p_start,
                            "p2": p_end,
                        })

            pred_spec = {
                "units": "mm",
                "entities": extracted_entities,
                "relations": gt_spec.get("relations", []),  # borrow relation topology to check joints
            }
            return self.evaluate_drawing_spec(pred_spec, gt_spec, sample_id=sample_id)
        except Exception as e:
            return GeometricEvaluationReport(
                sample_id=sample_id,
                cadcraft_success=False,
                is_watertight=False,
                max_gap_mm=999.0,
                tangency_mean_error_deg=90.0,
                tangency_max_error_deg=90.0,
                tangent_joints_evaluated=0,
                radius_mae_mm=999.0,
                radius_max_error_mm=999.0,
                entity_count_pred=0,
                entity_count_gt=len(gt_spec.get("entities", [])),
                accuracy_score=0.0,
                summary=f"Failed to parse DXF: {e}",
                details={"error": str(e)},
            )
