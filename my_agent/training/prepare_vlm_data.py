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
1. Decompose the drawing into distinct functional parts (base envelope, circular features/bores, cuts/joints, fillets, annotations).
2. Anchor strictly at origin (0.0, 0.0) - calculate all coordinates, centers, and dimensions relative to (0.0, 0.0).
3. For rounded corners and fillets: construct nominal sharp edges first, then apply fillet tools to solve exact tangent points and curve arcs.
4. For circle cuts and notches: compute exact circle-line intersection points and sweep kept arcs to form watertight closed joints.
5. Generate the executable CADCraft commands to build the exact watertight 2D model with dimensions and annotations."""


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


def convert_expert_dataset(expert_jsonl: str, output_dir: str):
    """Converts cadcraft_expert_qa.jsonl into Qwen, OpenAI, and Gemini SFT formats."""
    if not os.path.exists(expert_jsonl):
        print(f"[WARN] Expert dataset not found at {expert_jsonl}")
        return

    os.makedirs(output_dir, exist_ok=True)
    qwen_out = os.path.join(output_dir, "cadcraft_expert_qwen_sft.jsonl")
    openai_out = os.path.join(output_dir, "cadcraft_expert_openai_sft.jsonl")
    gemini_out = os.path.join(output_dir, "cadcraft_expert_gemini_sft.jsonl")

    count = 0
    with open(expert_jsonl, "r") as f_in, \
         open(qwen_out, "w") as f_qwen, \
         open(openai_out, "w") as f_openai, \
         open(gemini_out, "w") as f_gemini:
        for idx, line in enumerate(f_in):
            if not line.strip():
                continue
            data = json.loads(line)
            msgs = data["messages"]
            sys_msg = next((m["content"] for m in msgs if m["role"] == "system"), "")
            usr_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            ast_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")

            f_qwen.write(json.dumps({
                "id": f"cadcraft_expert_{idx+1:03d}",
                "conversations": [
                    {"from": "system", "value": sys_msg},
                    {"from": "user", "value": usr_msg},
                    {"from": "gpt", "value": ast_msg},
                ],
            }) + "\n")

            f_openai.write(json.dumps({
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
                    {"role": "assistant", "content": ast_msg},
                ],
            }) + "\n")

            f_gemini.write(json.dumps({
                "systemInstruction": {"parts": [{"text": sys_msg}]},
                "contents": [
                    {"role": "user", "parts": [{"text": usr_msg}]},
                    {"role": "model", "parts": [{"text": ast_msg}]},
                ],
            }) + "\n")
            count += 1

    print(f"[OK] Converted {count} CADCraft expert QA records into Qwen, OpenAI, and Gemini SFT formats in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare VLM training dataset")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Path to generated dataset")
    parser.add_argument("--expert_jsonl", type=str, default="./training/data/cadcraft_expert_qa.jsonl", help="Path to CADCraft expert QA jsonl")
    parser.add_argument("--output_dir", type=str, default="./training/data", help="Output directory for training files")
    parser.add_argument("--format", type=str, default="openai", choices=["openai", "qwen", "gemini"], help="Format type")
    parser.add_argument("--process_expert_only", action="store_true", help="Only process expert QA dataset")
    args = parser.parse_args()

    if args.expert_jsonl and os.path.exists(args.expert_jsonl):
        convert_expert_dataset(args.expert_jsonl, args.output_dir)

    if not args.process_expert_only and os.path.exists(os.path.join(args.dataset_dir, "manifest.json")):
        build_training_sets(args.dataset_dir, args.output_dir, format_type=args.format)


if __name__ == "__main__":
    main()

