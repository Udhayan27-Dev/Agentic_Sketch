"""CLI Runner to batch-generate and verify the CAD Technical Drawing Dataset.

Usage:
    python harness/generate_dataset.py --num_train 50 --num_val 10 --output_dir ./dataset
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.dataset_generator import (
    GeneratedSample,
    execute_sample_in_cadcraft,
    generate_sample_by_index,
    render_drawing_image,
)


def sample_to_drawing_spec_dict(sample: GeneratedSample) -> Dict[str, Any]:
    """Converts GeneratedSample to DrawingSpec schema matching sketchcad_agent/agent.py."""
    entities = []
    for e in sample.entities:
        ent_dict = {
            "id": e.id,
            "type": e.type,
        }
        if e.p1 is not None:
            ent_dict["p1"] = [round(x, 4) for x in e.p1]
        if e.p2 is not None:
            ent_dict["p2"] = [round(x, 4) for x in e.p2]
        if e.center is not None:
            ent_dict["center"] = [round(x, 4) for x in e.center]
        if e.radius is not None:
            ent_dict["radius"] = round(e.radius, 4)
        if e.start_deg is not None:
            ent_dict["start_deg"] = round(e.start_deg, 4)
        if e.end_deg is not None:
            ent_dict["end_deg"] = round(e.end_deg, 4)
        entities.append(ent_dict)

    relations = []
    for r in sample.relations:
        rel_dict = {
            "kind": r.kind,
            "refs": r.refs,
            "value_mm": round(r.value_mm, 4) if r.value_mm is not None else None,
        }
        relations.append(rel_dict)

    annotations = []
    for a in sample.annotations:
        ann_dict = {
            "text": a.text,
            "position": [round(x, 4) for x in a.position],
            "height": round(a.height, 4),
            "rotation_deg": round(a.rotation_deg, 4),
        }
        annotations.append(ann_dict)

    return {
        "units": "mm",
        "entities": entities,
        "relations": relations,
        "annotations": annotations,
        "assumptions": [
            "Origin (0.0, 0.0) is fixed at the base starting point; all coordinates are referenced from this origin.",
            f"Shape category: {sample.category}.",
        ],
        "unreadable": [],
    }


def generate_split(split_name: str, count: int, base_dir: str, verify_cadcraft: bool = True) -> list:
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    manifest_entries = []

    print(f"--> Generating {count} samples for split '{split_name}' in {split_dir}...")
    success_count = 0

    for i in range(count):
        sample = generate_sample_by_index(i, split=split_name)
        sid = sample.sample_id

        # 1. Render PNG Technical Drawing
        img_path = os.path.join(split_dir, f"{sid}_drawing.png")
        render_drawing_image(sample, img_path)

        # 2. Save DrawingSpec JSON
        spec_dict = sample_to_drawing_spec_dict(sample)
        spec_path = os.path.join(split_dir, f"{sid}_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec_dict, f, indent=2)

        # 3. Save CADCraft commands JSON
        cmds_path = os.path.join(split_dir, f"{sid}_commands.json")
        with open(cmds_path, "w") as f:
            json.dump(sample.cadcraft_commands, f, indent=2)

        # 4. Save CADCraft .scr script
        scr_path = os.path.join(split_dir, f"{sid}_script.scr")
        with open(scr_path, "w") as f:
            f.write(sample.cadcraft_script)

        # 5. Verify & Execute in CADCraft
        dxf_path = os.path.join(split_dir, f"{sid}.dxf")
        cadcraft_png = os.path.join(split_dir, f"{sid}_cadcraft.png")
        cadcraft_ok = False
        cadcraft_msg = ""

        if verify_cadcraft:
            cadcraft_ok, dxf_res, png_res = execute_sample_in_cadcraft(sample, split_dir)
            if cadcraft_ok:
                success_count += 1
            else:
                cadcraft_msg = png_res

        entry = {
            "sample_id": sid,
            "split": split_name,
            "category": sample.category,
            "description": sample.description,
            "drawing_image": os.path.relpath(img_path, base_dir),
            "drawing_spec": os.path.relpath(spec_path, base_dir),
            "cadcraft_commands": os.path.relpath(cmds_path, base_dir),
            "cadcraft_script": os.path.relpath(scr_path, base_dir),
            "cadcraft_dxf": os.path.relpath(dxf_path, base_dir) if cadcraft_ok else None,
            "cadcraft_png": os.path.relpath(cadcraft_png, base_dir) if cadcraft_ok else None,
            "cadcraft_verified": cadcraft_ok,
            "num_entities": len(sample.entities),
            "num_relations": len(sample.relations),
            "num_dimensions": len(sample.dimensions),
        }
        manifest_entries.append(entry)

        if (i + 1) % 10 == 0 or (i + 1) == count:
            print(f"    [{split_name}] Completed {i + 1}/{count} samples (CADCraft verified: {success_count}/{i + 1})")

    return manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Batch generate CAD technical drawing dataset")
    parser.add_argument("--num_train", type=int, default=50, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=10, help="Number of validation samples")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--skip_cadcraft", action="store_true", help="Skip cadcraft-cli verification")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================================")
    print(" SketchCAD Agent: CADCraft Dataset Generator Harness")
    print(f" Train: {args.num_train} | Val: {args.num_val} | Output: {output_dir}")
    print("==================================================================")

    train_entries = generate_split("train", args.num_train, output_dir, verify_cadcraft=not args.skip_cadcraft)
    val_entries = generate_split("val", args.num_val, output_dir, verify_cadcraft=not args.skip_cadcraft)

    manifest = {
        "num_train": len(train_entries),
        "num_val": len(val_entries),
        "total": len(train_entries) + len(val_entries),
        "categories": ["corner_fillets", "tangent_profile", "slotted_link", "keyway_cutout", "notched_flange"],
        "train": train_entries,
        "val": val_entries,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Manifest written to {manifest_path}")
    print(f"Total verified samples: {manifest['total']}")


if __name__ == "__main__":
    main()
