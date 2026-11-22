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


def test_analyst_agent_configuration_and_cad_directive_prompt():
    from sketchcad_agent.agent import analyst, ANALYST_PROMPT, draftsman_instruction, python_tool, add_fillet, read_entity, delete_entity, draftsman
    # Analyst outputs freeform prompt directive (no rigid schema failure)
    assert analyst.output_schema is None
    assert analyst.output_key == 'cad_description'
    assert python_tool in analyst.tools
    # Prompt embeds comprehensive CAD engineering knowledge
    assert "READ:" in ANALYST_PROMPT
    assert "IDENTIFY:" in ANALYST_PROMPT
    assert "DATUM:" in ANALYST_PROMPT
    assert "SOLVE:" in ANALYST_PROMPT
    assert "SELF-CHECK" in ANALYST_PROMPT
    assert "FILLETS:" in ANALYST_PROMPT
    assert "outer_loop" in ANALYST_PROMPT
    # Draftsman instruction binds drawing_spec
    assert "{drawing_spec}" in draftsman_instruction
    assert add_fillet in draftsman.tools
    assert read_entity in draftsman.tools
    assert delete_entity in draftsman.tools


@pytest.mark.anyio
async def test_sanitize_analyst_output_populates_state():
    from sketchcad_agent.agent import _sanitize_analyst_output
    from google.genai import types

    class DummyContext:
        def __init__(self):
            self.state = {}

    class DummyResponse:
        def __init__(self, text):
            self.content = types.Content(role='model', parts=[types.Part.from_text(text=text)])

    ctx = DummyContext()
    resp = DummyResponse("I am Qwen. # TECHNICAL CAD DRAFTING DIRECTIVE\n- Component: Cam Lever")
    await _sanitize_analyst_output(ctx, resp)

    assert resp.content.parts[0].text.startswith("# TECHNICAL CAD DRAFTING DIRECTIVE")
    assert ctx.state.get('cad_description') == resp.content.parts[0].text
    assert ctx.state.get('drawing_spec') == resp.content.parts[0].text


def test_python_tool_execution():
    from sketchcad_agent.agent import python_tool
    res1 = python_tool("round(2.3 * 12.0, 2)")
    assert "27.6" in res1
    res2 = python_tool("import math; print(round(math.sin(math.radians(90)), 2))")
    assert "1.0" in res2


def test_normalize_execute_args():
    from sketchcad_agent.agent import _normalize_execute_args
    # Test line normalization
    norm_line = _normalize_execute_args({'command': 'line', 'p1': [0, 0], 'p2': [10, 0]})
    assert norm_line['command'] == 'line'
    assert norm_line['params']['points'] == [[0, 0], [10, 0]]

    # Test arc normalization
    norm_arc = _normalize_execute_args({'command': 'arc', 'center': [5, 5], 'radius': 5, 'start_angle': 0, 'end_angle': 90})
    assert norm_arc['command'] == 'arc'
    assert norm_arc['params']['center'] == [5, 5]
    assert norm_arc['params']['start'] == 0
    assert norm_arc['params']['end'] == 90


def test_analyst_json_spec_validation():
    from sketchcad_agent.agent import AnalystJsonSpec, BoundingBox, OuterLoopLine, OuterLoopArc, InnerCircle, FilletSpec, SelfChecks
    spec_data = {
        "units": "mm",
        "h": 12.0,
        "origin": "baseline, bottom-left of bounding box",
        "bbox": {"xmin": 0.0, "xmax": 27.6, "ymin": 0.0, "ymax": 12.0},
        "stated_labels": ["h=12", "R1=0.625h", "2.3h"],
        "outer_loop": [
            {"id": "L1", "type": "line", "p1": [1.5, 0.0], "p2": [27.6, 0.0]},
            {"id": "L2", "type": "line", "p1": [27.6, 0.0], "p2": [24.0, 3.0]},
            {"id": "A1", "type": "arc", "center": [7.5, 4.5], "radius": 7.5, "start_angle": 180.0, "end_angle": 336.4, "sweep": 156.4, "midpoint": [0.62, 1.48], "p_start": [0.0, 4.5], "p_end": [14.37, 1.48]}
        ],
        "inner_features": [
            {"id": "C1", "type": "circle", "center": [7.5, 4.5], "radius": 3.6}
        ],
        "fillets": [
            {"corner": [24.0, 3.0], "between": ["L2", "L3"], "radius": 0.5, "basis": "ESTIMATED", "expected_sweep": 90.0}
        ],
        "checks": {"loop_closed": True, "bbox_ok": True, "arcs_ok": True},
        "uncertain": []
    }
    model = AnalystJsonSpec.model_validate(spec_data)
    assert model.units == "mm"
    assert model.h == 12.0
    assert len(model.outer_loop) == 3
    assert model.outer_loop[0].id == "L1"
    assert model.outer_loop[2].type == "arc"
    assert len(model.inner_features) == 1
    assert model.fillets[0].basis == "ESTIMATED"
    assert model.checks.loop_closed is True


@pytest.mark.anyio
async def test_cadcraft_draftsman_tools():
    from sketchcad_agent.agent import active_cad, read_entity, delete_entity, add_fillet, _entity_id_map
    _entity_id_map.clear()
    await active_cad._execute_with_session(lambda s: s.call_tool('new_drawing', {'metric': True}), 'Error in new_drawing')

    # Draw two lines
    r1 = await active_cad._execute_with_session(lambda s: s.call_tool('execute', {'command': 'line', 'params': {'points': [[0, 0], [10, 0]]}}), 'Error in line 1')
    l1 = await read_entity('L1')
    assert l1['type'] == 'line'
    assert l1['p1'] == [0.0, 0.0]
    assert l1['p2'] == [10.0, 0.0]

    r2 = await active_cad._execute_with_session(lambda s: s.call_tool('execute', {'command': 'line', 'params': {'points': [[10, 0], [10, 10]]}}), 'Error in line 2')
    l2 = await read_entity('L2')
    assert l2['type'] == 'line'
    assert l2['p1'] == [10.0, 0.0]

    # Test add_fillet with oversize radius (recovery check)
    fail_res = await add_fillet('L1', 'L2', 15.0)
    assert fail_res['status'] == 'failed'
    assert 'Halve the radius' in fail_res['error']

    # Test add_fillet with valid radius
    ok_res = await add_fillet('L1', 'L2', 2.0)
    assert ok_res['status'] == 'ok'
    assert 'fillet_arc' in ok_res
    arc = ok_res['fillet_arc']
    assert arc['radius'] == 2.0
    assert arc['sweep'] < 180.0

    # Test delete_entity
    del_res = await delete_entity('L1')
    assert del_res['status'] == 'deleted'


@pytest.mark.anyio
async def test_sanitize_analyst_output_strips_tool_call_when_spec_ready():
    from sketchcad_agent.agent import _sanitize_analyst_output
    from google.genai import types

    class DummyContext:
        def __init__(self):
            self.state = {}

    json_text = (
        "WORKING\nComputing geometry with python_tool\n"
        "{\n"
        '  "units": "mm",\n'
        '  "h": 12,\n'
        '  "origin": "baseline, bottom-left of bounding box",\n'
        '  "bbox": {"xmin": 0, "xmax": 27.6, "ymin": 0, "ymax": 12},\n'
        '  "stated_labels": ["h=12"],\n'
        '  "outer_loop": [{"id": "L1", "type": "line", "p1": [0,0], "p2": [10,0]}],\n'
        '  "inner_features": [],\n'
        '  "fillets": [],\n'
        '  "checks": {"loop_closed": true, "bbox_ok": true, "arcs_ok": true},\n'
        '  "uncertain": []\n'
        "}"
    )

    class DummyResponseWithToolCall:
        def __init__(self):
            self.content = types.Content(
                role='model',
                parts=[
                    types.Part.from_text(text=json_text),
                    types.Part.from_function_call(name='python_tool', args={'code': '2+2'})
                ]
            )

    ctx = DummyContext()
    resp = DummyResponseWithToolCall()
    assert any(getattr(p, 'function_call', None) for p in resp.content.parts)

    await _sanitize_analyst_output(ctx, resp)

    # Tool calls must be stripped to prevent runaway loops!
    assert not any(getattr(p, 'function_call', None) for p in resp.content.parts)
    assert len(resp.content.parts) == 1
    assert "outer_loop" in ctx.state.get('drawing_spec', '')


def test_extract_valid_json_spec_with_multiple_concatenated_jsons():
    from sketchcad_agent.agent import _extract_valid_json_spec
    json_block = '{"units": "mm", "bbox": {"xmin": 0, "xmax": 10, "ymin": 0, "ymax": 10}, "outer_loop": []}'
    repeated_text = f"Working notes... {json_block} and then repeated {json_block}"
    parsed, cand = _extract_valid_json_spec(repeated_text)
    assert parsed is not None
    assert parsed['units'] == 'mm'
    assert parsed['bbox']['xmax'] == 10


