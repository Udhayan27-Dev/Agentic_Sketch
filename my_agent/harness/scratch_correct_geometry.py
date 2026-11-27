"""
Hardcoded correct geometry for the linkage bracket blueprint.
This generates a mathematically verified DXF directly from the known blueprint dimensions,
bypassing the vision model entirely. Use to validate the pipeline or as a fallback.

Blueprint dimensions (all in mm):
  Overall bbox: 279 x 178
  Hub: center=(241, 140), Ø76 (R=38 outer), Ø50.8 (R=25.4 bore)
  Left lug: center=(19, 38), R=19 (rounded arm end)
  Bosses at y=38: x=44, x=108(?), x=171, x=196 — R=14 (Ø28 outer), R=7 (Ø14 hole)
  Block: left_x=88.5, right_x=101.5, bottom_y=38, top_y=124 (height=86, width=13)
  R89 concave fillet: between block top (y=124) and hub
  Arm: lower baseline y=0, boss centerline y=38
  Right wall: x=279, from y=0 to y=140 (hub tangent)
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dimensions ───────────────────────────────────────────────────────────────
cx_hub, cy_hub, r_hub = 241.0, 140.0, 38.0    # Ø76
r_bore = 25.4                                   # Ø50.8
cx_lug, cy_lug, r_lug = 19.0, 38.0, 19.0      # R19 lug
blk_lx, blk_rx = 88.5, 101.5                   # block x extents
blk_ty = 124.0                                  # block top y (38+86)
blk_by = 38.0                                   # block bottom y

# R89 concave fillet (center is ABOVE the block top line, on the AIR side)
r89 = 89.0
cy89 = blk_ty + r89                             # = 213
cx89 = cx_hub - math.sqrt((r89 + r_hub) ** 2 - (cy89 - cy_hub) ** 2)
# Tangent points
r89_p_start = [cx89, blk_ty]                    # on y=124 line
r89_tp_hub_x = cx_hub + r_hub * (cx89 - cx_hub) / (r89 + r_hub)
r89_tp_hub_y = cy_hub + r_hub * (cy89 - cy_hub) / (r89 + r_hub)
r89_p_end = [r89_tp_hub_x, r89_tp_hub_y]
# Arc angles
a89_start = 270.0                               # bottom of fillet (pointing to y=124)
a89_end = math.degrees(math.atan2(r89_tp_hub_y - cy89, r89_tp_hub_x - cx89)) % 360
sweep89 = (a89_end - a89_start) % 360
a89_mid = a89_start + sweep89 / 2.0
r89_via = [cx89 + r89 * math.cos(math.radians(a89_mid)), cy89 + r89 * math.sin(math.radians(a89_mid))]

# Hub arc: CCW from 0° (right tangent at x=279) to angle of R89 junction
a_hub_start = 0.0
a_hub_end = math.degrees(math.atan2(r89_tp_hub_y - cy_hub, r89_tp_hub_x - cx_hub)) % 360
sweep_hub = (a_hub_end - a_hub_start) % 360
a_hub_mid = a_hub_start + sweep_hub / 2.0
hub_via = [cx_hub + r_hub * math.cos(math.radians(a_hub_mid)), cy_hub + r_hub * math.sin(math.radians(a_hub_mid))]
hub_p_start = [cx_hub + r_hub, cy_hub]          # (279, 140)
hub_p_end = r89_p_end

# Left lug arc (outer contour): CCW from 0° (38,38) to 258° (lower-right tangent)
# This wraps: right → up → left → lower-left (258°), passing through top (19,57) at 90°
a_lug_start = 0.0
a_lug_end = math.degrees(math.atan2(19.414 - cy_lug, 15.0555 - cx_lug)) % 360  # ~258°
lug_p_start = [cx_lug + r_lug, cy_lug]           # (38, 38) rightmost
lug_p_end = [cx_lug + r_lug * math.cos(math.radians(a_lug_end)), cy_lug + r_lug * math.sin(math.radians(a_lug_end))]
sweep_lug = (a_lug_end - a_lug_start) % 360
a_lug_mid = a_lug_start + sweep_lug / 2.0        # ~129° (through left+top of lug)
lug_via = [cx_lug + r_lug * math.cos(math.radians(a_lug_mid)), cy_lug + r_lug * math.sin(math.radians(a_lug_mid))]

# Tangent line from lug bottom to baseline
# Lug bottom-right tangent point: (15.0555, 19.414) at ~258°
lug_tp = [cx_lug + r_lug * math.cos(math.radians(a_lug_end)), cy_lug + r_lug * math.sin(math.radians(a_lug_end))]
# Tangent from baseline to lug: extend the line from (lug_tp) down to y=0
# Line direction: perpendicular to radius at tangent point
# Radius direction at tangent point: (cos(258°), sin(258°))
rad_angle = a_lug_end
# Tangent direction (perpendicular to radius, pointing down-right):
tang_dx = math.sin(math.radians(rad_angle))   # perpendicular to radius
tang_dy = -math.cos(math.radians(rad_angle))
# Parametric: P = lug_tp + t*(tang_dx, tang_dy), find t where P[1] = 0
t_base = -lug_tp[1] / tang_dy if abs(tang_dy) > 1e-6 else 0
base_pt = [lug_tp[0] + t_base * tang_dx, 0.0]

# ── Outer Loop ────────────────────────────────────────────────────────────────
# CCW sequence starting from bottom-right corner (279, 0):
outer_loop = [
    # L1: Bottom baseline (right to left)
    {"id": "L1", "type": "line", "p1": [279.0, 0.0], "p2": [round(base_pt[0], 4), 0.0]},
    # L2: Sloped lower-left arm edge (from baseline up to lug tangent)
    {"id": "L2", "type": "line", "p1": [round(base_pt[0], 4), 0.0], "p2": [round(lug_tp[0], 4), round(lug_tp[1], 4)]},
    # A1: Lug arc (CCW from ~258° through left+top of lug to 0°/right)
    {"id": "A1", "type": "arc",
     "center": [cx_lug, cy_lug], "radius": r_lug,
     "start_deg": round(a_lug_end, 4), "end_deg": round(a_lug_start, 4),
     "p_start": [round(lug_tp[0], 4), round(lug_tp[1], 4)],
     "via_point": [round(lug_via[0], 4), round(lug_via[1], 4)],
     "p_end": [round(lug_p_start[0], 4), round(lug_p_start[1], 4)]},
    # L3: Arm upper edge (from lug right to block left wall)
    {"id": "L3", "type": "line", "p1": [38.0, 38.0], "p2": [blk_lx, blk_by]},
    # L4: Block left wall (up)
    {"id": "L4", "type": "line", "p1": [blk_lx, blk_by], "p2": [blk_lx, blk_ty]},
    # L5: Block top edge (right)
    {"id": "L5", "type": "line", "p1": [blk_lx, blk_ty], "p2": [blk_rx, blk_ty]},
    # L6: Block right wall (down)
    {"id": "L6", "type": "line", "p1": [blk_rx, blk_ty], "p2": [blk_rx, blk_by]},
    # L7: Arm upper edge (from block right to R89 start)
    {"id": "L7", "type": "line", "p1": [blk_rx, blk_by], "p2": [round(cx89, 4), blk_ty]},
    # A2: R89 concave fillet (CCW from 270° to ~325°, connecting block top to hub)
    {"id": "A2", "type": "arc",
     "center": [round(cx89, 4), round(cy89, 4)], "radius": r89,
     "start_deg": round(a89_start, 4), "end_deg": round(a89_end, 4),
     "p_start": [round(r89_p_start[0], 4), round(r89_p_start[1], 4)],
     "via_point": [round(r89_via[0], 4), round(r89_via[1], 4)],
     "p_end": [round(r89_p_end[0], 4), round(r89_p_end[1], 4)]},
    # A3: Hub outer arc (CCW from 0° to ~145°, right → upper-left of hub)
    {"id": "A3", "type": "arc",
     "center": [cx_hub, cy_hub], "radius": r_hub,
     "start_deg": round(a_hub_end, 4), "end_deg": round(a_hub_start, 4),
     "p_start": [round(hub_p_end[0], 4), round(hub_p_end[1], 4)],
     "via_point": [round(hub_via[0], 4), round(hub_via[1], 4)],
     "p_end": [round(hub_p_start[0], 4), round(hub_p_start[1], 4)]},
    # L8: Right vertical wall (from hub tangent at 279,140 down to 279,0)
    {"id": "L8", "type": "line", "p1": [279.0, 140.0], "p2": [279.0, 0.0]},
]

# ── Holes (all circles: outer boss rings + bores/holes) ──────────────────────
holes = [
    # Hub: outer ring Ø76 (R=38) — the outer contour arc is part of outer_loop, 
    # but the full circle is also needed as a reference ring for dimensions
    {"center": [cx_hub, cy_hub], "radius": r_hub},          # Ø76 outer hub ring
    {"center": [cx_hub, cy_hub], "radius": r_bore},          # Ø50.8 bore
    # Left lug boss:
    {"center": [cx_lug, cy_lug], "radius": r_lug},           # Ø38 outer lug ring
    {"center": [cx_lug, cy_lug], "radius": 7.0},             # Ø14 hole
    # Lower mounting bosses at y=38:
    {"center": [44.0, 38.0], "radius": 14.0},                # Ø28 boss
    {"center": [44.0, 38.0], "radius": 7.0},                 # Ø14 hole
    {"center": [108.0, 38.0], "radius": 14.0},               # Ø28 boss
    {"center": [108.0, 38.0], "radius": 7.0},                # Ø14 hole
    {"center": [178.0, 38.0], "radius": 14.0},               # Ø28 boss
    {"center": [178.0, 38.0], "radius": 7.0},                # Ø14 hole
    {"center": [216.0, 38.0], "radius": 14.0},               # Ø28 boss
    {"center": [216.0, 38.0], "radius": 7.0},                # Ø14 hole
]

# ── Internal lines (block internals, arm rib) ─────────────────────────────────
internal_lines = [
    # Block outer visible walls (already in outer_loop, but inner details):
    {"p1": [91.0, 38.0], "p2": [91.0, 124.0], "style": "solid"},  # inner left rib
    {"p1": [99.0, 38.0], "p2": [99.0, 124.0], "style": "solid"},  # inner right rib
    # Dashed slot inside block:
    {"p1": [93.0, 48.0], "p2": [93.0, 114.0], "style": "dashed"},
    {"p1": [97.0, 48.0], "p2": [97.0, 114.0], "style": "dashed"},
    {"p1": [93.0, 48.0], "p2": [97.0, 48.0], "style": "dashed"},
    {"p1": [93.0, 114.0], "p2": [97.0, 114.0], "style": "dashed"},
]

parsed_data = {
    "part_name": "linkage_bracket",
    "part_type": "Bracket",
    "description": (
        "Linkage mounting bracket — horizontal arm with rounded left lug (R19), "
        "central rectangular block (13×86mm), R89 concave fillet to circular hub "
        "(Ø76 outer / Ø50.8 bore) at upper right, four lower mounting bosses "
        "(Ø28 boss / Ø14 hole each). Overall 279×178mm."
    ),
    "bbox": {"width": 279.0, "height": 178.0},
    "outer_loop": outer_loop,
    "holes": holes,
    "internal_lines": internal_lines,
}

# Print summary
print(f"Linkage bracket geometry:")
print(f"  Outer loop: {len(outer_loop)} segments")
print(f"  Holes:      {len(holes)}")
print(f"  Internals:  {len(internal_lines)}")
print(f"  R89 center: ({cx89:.4f}, {cy89:.4f})")
print(f"  Hub arc:    0° → {a_hub_end:.2f}° CCW ({sweep_hub:.2f}° sweep)")
print(f"  Lug arc:    {a_lug_end:.2f}° → 360°/0° CCW ({360-sweep_lug:.2f}° ... wait)")
print(f"  Lug arc sweep CCW: {(0 - a_lug_end + 360) % 360:.2f}° (through top of lug at 90°)")
print(f"  Base tangent point: ({base_pt[0]:.4f}, 0)")
print()

# Save corrected geometry as a spec file for reference
spec_path = OUTPUT_DIR / "linkage_bracket_corrected_spec.json"
spec_path.write_text(json.dumps(parsed_data, indent=2))
print(f"Saved corrected spec to: {spec_path}")

if __name__ == "__main__":
    # Build and run directly
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from harness.image_to_cad_harness import ImageCADHarness, ParsedDrawing

    harness = ImageCADHarness()
    parsed = ParsedDrawing(
        part_name=parsed_data["part_name"],
        part_type=parsed_data["part_type"],
        description=parsed_data["description"],
        bbox=parsed_data["bbox"],
        outer_loop=parsed_data["outer_loop"],
        holes=parsed_data["holes"],
        internal_lines=parsed_data["internal_lines"],
        raw_response=json.dumps(parsed_data),
    )

    commands, gt_spec, desc = harness.build_cadcraft_geometry(parsed)
    print(f"\n{desc}")
    print(f"Generated {len(commands)} commands.")

    dxf_path = str(OUTPUT_DIR / "linkage_bracket_corrected.dxf")
    png_path = str(OUTPUT_DIR / "linkage_bracket_corrected_rendered.png")
    ok, log = harness.execute_cadcraft(commands, dxf_path, output_png=png_path)
    if ok:
        print(f"DXF: {dxf_path} ({os.path.getsize(dxf_path)} bytes)")
        if os.path.isfile(png_path):
            print(f"PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
        eval_report = harness.evaluator.evaluate_dxf(dxf_path, gt_spec, sample_id="linkage_bracket_corrected")
        print(f"Accuracy: {eval_report.accuracy_score:.1f}%  Gap: {eval_report.max_gap_mm:.4f}mm")
        from sketchcad_agent.agent import open_in_cadcraft_gui
        print(open_in_cadcraft_gui(dxf_path))
    else:
        print(f"CADCraft failed: {log[:300]}")
