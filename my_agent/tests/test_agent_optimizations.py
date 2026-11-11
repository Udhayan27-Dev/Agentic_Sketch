import io
import json
import pytest
from PIL import Image
from google.genai import types
from google.adk.models import LlmRequest
from sketchcad_agent.agent import (
    sanitize_json_output,
    _detect_and_summarize_image,
    _pace_and_preprocess,
    DrawingSpec,
    CriticReport,
    active_model
)


def test_sanitize_json_output_with_persona_greeting():
    raw_text = 'I am Qwen. {"units": "mm", "entities": [{"id": "e1", "type": "line", "p1": [0.0, 0.0], "p2": [65.0, 0.0]}], "relations": [], "annotations": [], "assumptions": ["test"], "unreadable": []}'
    sanitized = sanitize_json_output(raw_text, default_schema=DrawingSpec)
    parsed = DrawingSpec.model_validate_json(sanitized)
    assert parsed.units == "mm"
    assert len(parsed.entities) == 1
    assert parsed.entities[0].id == "e1"


def test_sanitize_json_output_with_markdown_fences():
    raw_text = '```json\n{"units": "mm", "entities": [], "relations": [], "assumptions": [], "unreadable": []}\n```'
    sanitized = sanitize_json_output(raw_text, default_schema=DrawingSpec)
    parsed = DrawingSpec.model_validate_json(sanitized)
    assert parsed.units == "mm"


def test_sanitize_json_output_empty_fallback():
    raw_text = "I am Qwen. I cannot generate drawing spec."
    sanitized = sanitize_json_output(raw_text, default_schema=DrawingSpec)
    parsed = DrawingSpec.model_validate_json(sanitized)
    assert parsed.units == "mm"
    assert len(parsed.entities) >= 1


def test_detect_and_summarize_image_plug_gauge():
    im = Image.new('RGB', (300, 150), color='white')
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    summary = _detect_and_summarize_image(buf.getvalue())
    assert "Double-Ended Plain Cylindrical Limit Plug Gauge" in summary
    assert "9.985" in summary


@pytest.mark.anyio
async def test_pace_and_preprocess_replaces_image(monkeypatch):
    im = Image.new('RGB', (200, 100), color='white')
    buf = io.BytesIO()
    im.save(buf, format='PNG')

    # Test multimodal mode: preserves image part
    monkeypatch.setenv('SKETCHCAD_MULTIMODAL', '1')
    image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type='image/png')
    req = LlmRequest(model='openai/qwen3.8-27b', contents=[types.Content(role='user', parts=[image_part])])
    await _pace_and_preprocess(None, req)
    assert len(req.contents[0].parts) == 1
    assert req.contents[0].parts[0].inline_data is not None

    # Test fallback mode: converts to text summary
    monkeypatch.setenv('SKETCHCAD_MULTIMODAL', '0')
    image_part2 = types.Part.from_bytes(data=buf.getvalue(), mime_type='image/png')
    req2 = LlmRequest(model='openai/qwen3.8-27b', contents=[types.Content(role='user', parts=[image_part2])])
    await _pace_and_preprocess(None, req2)
    assert len(req2.contents[0].parts) == 1
    first_part = req2.contents[0].parts[0]
    assert first_part.text is not None
    assert "DRAWING SPECIFICATION EXTRACTED FROM IMAGE" in first_part.text
    assert first_part.inline_data is None
