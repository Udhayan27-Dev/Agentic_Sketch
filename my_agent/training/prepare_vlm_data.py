"""Prepares multimodal vision-language fine-tuning datasets from the CADCraft harness.

Supports output formats:
1. 'qwen': Qwen2.5-VL / Qwen-VL conversational format.
2. 'openai': OpenAI Vision Fine-Tuning JSONL format (data:image/png;base64).
3. 'gemini': Google Vertex / GenAI SFT tuning format.
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SYSTEM_PROMPT = """You are SketchCAD Draftsman, an expert AI CAD engineer.
Given a 2D technical drawing image:
1. Transcribe the geometry into a structured DrawingSpec with exact coordinates in millimeters referenced from origin (0.0, 0.0).
2. For rounded corners and tangent blends, use exact geometric calculations for arcs and tangent lines.
3. Generate the executable CADCraft commands to build the exact watertight 2D model with dimensions and annotations."""


def encode_image_base64(image_path: str) -> str:
    """Encodes an image to base64 data URI."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def format_assistant_response(spec_dict: Dict[str, Any], cadcraft_cmds: List[Dict[str, Any]], script_text: str) -> str:
    """Creates a structured response with DrawingSpec and CADCraft commands."""
    res = {
        "drawing_spec": spec_dict,
        "cadcraft_commands": cadcraft_cmds,
        "cadcraft_script": script_text,
    }
    return json.dumps(res, indent=2)


def convert_sample(sample_entry: Dict[str, Any], dataset_dir: str, format_type: str = "openai") -> Dict[str, Any]:
    """Converts a single dataset manifest entry into target training format."""
    img_rel = sample_entry["drawing_image"]
    spec_rel = sample_entry["drawing_spec"]
    cmds_rel = sample_entry["cadcraft_commands"]
    scr_rel = sample_entry["cadcraft_script"]

    img_path = os.path.join(dataset_dir, img_rel)
    spec_path = os.path.join(dataset_dir, spec_rel)
    cmds_path = os.path.join(dataset_dir, cmds_rel)
    scr_path = os.path.join(dataset_dir, scr_rel)

    with open(spec_path, "r") as f:
        spec_dict = json.load(f)
    with open(cmds_path, "r") as f:
        cmds_list = json.load(f)
    with open(scr_path, "r") as f:
        scr_text = f.read()

    user_query = (
        f"Draw this {sample_entry['category']} technical drawing ({sample_entry['description']}) "
        "in CADCraft. Anchor strictly at origin (0.0, 0.0). Ensure exact tangency, watertight contours, "
        "and complete dimensional callouts."
    )
    assistant_content = format_assistant_response(spec_dict, cmds_list, scr_text)

    if format_type == "openai":
        b64_uri = encode_image_base64(img_path)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image_url", "image_url": {"url": b64_uri}},
                    ],
                },
                {"role": "assistant", "content": assistant_content},
            ]
        }
    elif format_type == "qwen":
        return {
            "id": sample_entry["sample_id"],
            "image": os.path.abspath(img_path),
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "user", "value": f"<image>\n{user_query}"},
                {"from": "gpt", "value": assistant_content},
            ],
        }
    elif format_type == "gemini":
        with open(img_path, "rb") as f:
            raw_b64 = base64.b64encode(f.read()).decode("utf-8")
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_query},
                        {"inlineData": {"mimeType": "image/png", "data": raw_b64}},
                    ],
                },
                {"role": "model", "parts": [{"text": assistant_content}]},
            ],
        }
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def build_training_sets(dataset_dir: str, output_dir: str, format_type: str = "openai"):
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run generate_dataset.py first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for split in ("train", "val"):
        samples = manifest.get(split, [])
        out_file = os.path.join(output_dir, f"{split}_{format_type}.jsonl")
        print(f"--> Writing {len(samples)} samples to {out_file} ({format_type} format)...")
        with open(out_file, "w") as f_out:
            for s in samples:
                row = convert_sample(s, dataset_dir, format_type=format_type)
                f_out.write(json.dumps(row) + "\n")
        print(f"    Done: {out_file} ({os.path.getsize(out_file)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Prepare VLM training dataset")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to generated dataset")
    parser.add_argument("--output_dir", type=str, default="./training/data", help="Output directory for training files")
    parser.add_argument("--format", type=str, default="openai", choices=["openai", "qwen", "gemini"], help="Format type")
    args = parser.parse_args()

    build_training_sets(args.dataset_dir, args.output_dir, format_type=args.format)


if __name__ == "__main__":
    main()
