#!/usr/bin/env python3
"""
Generates production-grade Obsidian-compatible Excalidraw architecture diagrams
for SketchCAD:
1. sketchcad_hybrid_cloud_architecture.excalidraw.md
2. sketchcad_agent_pipeline_lifecycle.excalidraw.md
"""

import json
import os
import sys

def create_element_id(prefix, idx):
    return f"{prefix}_{idx:04d}"

class ExcalidrawSceneBuilder:
    def __init__(self, title):
        self.title = title
        self.elements = []
        self.text_elements_md = []
        self._elem_count = 0

    def _next_id(self, prefix="elem"):
        self._elem_count += 1
        return create_element_id(prefix, self._elem_count)

    def add_group_box(self, x, y, w, h, label, bg_color="#f8f9fa", stroke_color="#adb5bd", stroke_style="dashed"):
        box_id = self._next_id("grp_box")
        title_id = self._next_id("grp_txt")
        
        box = {
            "id": box_id,
            "type": "rectangle",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": stroke_color,
            "backgroundColor": bg_color,
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": stroke_style,
            "roughness": 0,
            "opacity": 70,
            "groupIds": [],
            "roundness": {"type": 3},
            "seed": hash(box_id) % 1000000,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": [],
            "updated": 1,
            "link": None,
            "locked": False
        }
        self.elements.append(box)

        txt_elem = {
            "id": title_id,
            "type": "text",
            "x": x + 16,
            "y": y + 12,
            "width": w - 32,
            "height": 24,
            "angle": 0,
            "strokeColor": stroke_color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": None,
            "seed": hash(title_id) % 1000000,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            "text": label,
            "fontSize": 16,
            "fontFamily": 1,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "originalText": label,
            "lineHeight": 1.25
        }
        self.elements.append(txt_elem)
        self.text_elements_md.append(f"{label} ^{title_id}")
        return box_id

    def add_node(self, x, y, w, h, label, bg_color="#e7f5ff", stroke_color="#1971c2", text_color="#1864ab", font_size=15):
        rect_id = self._next_id("node")
        text_id = f"{rect_id}_txt"

        rect = {
            "id": rect_id,
            "type": "rectangle",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": stroke_color,
            "backgroundColor": bg_color,
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": {"type": 3},
            "seed": hash(rect_id) % 1000000,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": [{"id": text_id, "type": "text"}],
            "updated": 1,
            "link": None,
            "locked": False
        }
        self.elements.append(rect)

        lines = label.split("\n")
        line_height = font_size * 1.3
        total_text_h = len(lines) * line_height
        text_y = y + (h - total_text_h) / 2

        txt_elem = {
            "id": text_id,
            "type": "text",
            "x": x + 10,
            "y": max(y + 6, text_y),
            "width": w - 20,
            "height": total_text_h,
            "angle": 0,
            "strokeColor": text_color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 1,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": None,
            "seed": hash(text_id) % 1000000,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            "text": label,
            "fontSize": font_size,
            "fontFamily": 1,
            "textAlign": "center",
            "verticalAlign": "middle",
            "containerId": rect_id,
            "originalText": label,
            "lineHeight": 1.25
        }
        self.elements.append(txt_elem)
        self.text_elements_md.append(f"{label} ^{text_id}")
        return rect_id

    def add_arrow(self, start_x, start_y, end_x, end_y, label="", stroke_color="#495057", start_id=None, end_id=None):
        arrow_id = self._next_id("arr")
        points = [[0, 0], [end_x - start_x, end_y - start_y]]

        arrow = {
            "id": arrow_id,
            "type": "arrow",
            "x": start_x,
            "y": start_y,
            "width": end_x - start_x,
            "height": end_y - start_y,
            "angle": 0,
            "strokeColor": stroke_color,
            "backgroundColor": "transparent",
            "fillStyle": "solid",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 0,
            "opacity": 100,
            "groupIds": [],
            "roundness": {"type": 2},
            "seed": hash(arrow_id) % 1000000,
            "version": 1,
            "versionNonce": 1,
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
            "points": points,
            "lastCommittedPoint": None,
            "startBinding": {"elementId": start_id, "focus": 0, "gap": 5} if start_id else None,
            "endBinding": {"elementId": end_id, "focus": 0, "gap": 5} if end_id else None,
            "startArrowhead": None,
            "endArrowhead": "arrow"
        }
        self.elements.append(arrow)

        if label:
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2 - 18
            txt_id = f"{arrow_id}_txt"
            label_elem = {
                "id": txt_id,
                "type": "text",
                "x": mid_x - 70,
                "y": mid_y,
                "width": 140,
                "height": 20,
                "angle": 0,
                "strokeColor": stroke_color,
                "backgroundColor": "#ffffff",
                "fillStyle": "solid",
                "strokeWidth": 1,
                "strokeStyle": "solid",
                "roughness": 0,
                "opacity": 100,
                "groupIds": [],
                "roundness": None,
                "seed": hash(txt_id) % 1000000,
                "version": 1,
                "versionNonce": 1,
                "isDeleted": False,
                "boundElements": None,
                "updated": 1,
                "link": None,
                "locked": False,
                "text": label,
                "fontSize": 12,
                "fontFamily": 1,
                "textAlign": "center",
                "verticalAlign": "middle",
                "containerId": None,
                "originalText": label,
                "lineHeight": 1.2
            }
            self.elements.append(label_elem)
            self.text_elements_md.append(f"{label} ^{txt_id}")
        return arrow_id

    def build_markdown(self):
        scene = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff"
            },
            "files": {}
        }
        json_str = json.dumps(scene, indent=2)

        md = f"""---
excalidraw-plugin: parsed
tags: [excalidraw]
---
==⚠  Switch to EXCALIDRAW VIEW in the MORE OPTIONS menu of this document. ⚠==

# Excalidraw Data

## Text Elements
{chr(10).join(self.text_elements_md)}

%%
## Drawing
```json
{json_str}
```
%%
"""
        return md


def generate_architecture_diagram():
    b = ExcalidrawSceneBuilder("SketchCAD Hybrid Cloud-to-DGX Production Architecture")

    # Title header
    b.add_node(300, 20, 800, 50, "SketchCAD: Distributed Hybrid Production Architecture (AWS EC2 + NVIDIA DGX)", bg_color="#1864ab", stroke_color="#0b7285", text_color="#ffffff", font_size=18)

    # 1. Client Tier (Left)
    b.add_group_box(40, 90, 260, 680, "CLIENT TIER (Web & Mobile Browsers)", bg_color="#e7f5ff", stroke_color="#1971c2")
    n_user = b.add_node(60, 150, 220, 80, "End Users / Engineers\n(Upload Drawing PNG/PDF)\n(Inspect 2D Canvas CAD)", bg_color="#ffffff", stroke_color="#1971c2", text_color="#1864ab")
    n_react = b.add_node(60, 270, 220, 110, "React / Next.js Web App\n- Drag & Drop Uploader\n- Three.js / Canvas DXF Viewer\n- Live Agent Execution Stepper\n- DXF / SVG / STEP Downloads", bg_color="#d0ebff", stroke_color="#1864ab", text_color="#1864ab", font_size=13)
    n_ws_client = b.add_node(60, 430, 220, 80, "WebSocket / SSE Listener\nReal-time Critic Score &\nRefinement Logs Streaming", bg_color="#e7f5ff", stroke_color="#1971c2", text_color="#1864ab", font_size=13)
    n_auth = b.add_node(60, 550, 220, 80, "User Auth & Projects\n(JWT / OAuth2 / API Keys)\nProject History & Revisions", bg_color="#d0ebff", stroke_color="#1864ab", text_color="#1864ab", font_size=13)

    # 2. AWS Cloud Tier (Middle)
    b.add_group_box(340, 90, 440, 680, "AWS PUBLIC CLOUD TIER (EC2, S3, SQS, RDS)", bg_color="#fff4e6", stroke_color="#f76707")
    n_alb = b.add_node(370, 140, 380, 60, "Application Load Balancer (ALB) + SSL / TLS 1.3\nRoute 53 DNS & CloudFront Edge Cache", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    n_fastapi = b.add_node(370, 230, 380, 110, "AWS EC2: FastAPI Production Backend\n- Nginx Reverse Proxy + Gunicorn/Uvicorn Workers\n- Presigned S3 URL Generation\n- Job Dispatcher & Rate Limiter (Redis)\n- WebSocket Manager for Live Progress", bg_color="#ffffff", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    
    n_s3 = b.add_node(370, 370, 180, 100, "Amazon S3\nObject Storage\n- /uploads (PNG/PDF)\n- /artifacts (DXF/SVG)\n- /audits (Critic JSON)", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    n_sqs = b.add_node(570, 370, 180, 100, "Amazon SQS\nAsynchronous Queue\n- FIFO Job Queue\n- Dead-Letter Queue (DLQ)\n- Priority Channels", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)

    n_rds = b.add_node(370, 500, 180, 80, "PostgreSQL RDS\n- Jobs & Revisions\n- Critic Scores & Audits\n- User Accounts", bg_color="#ffffff", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    n_redis = b.add_node(570, 500, 180, 80, "Redis Cluster\n- Pub/Sub SSE Bridge\n- Rate Limiting\n- Session Tokens", bg_color="#ffffff", stroke_color="#e8590c", text_color="#d9480f", font_size=13)

    n_cloudwatch = b.add_node(370, 610, 380, 60, "CloudWatch + Prometheus / Grafana\nTelemetry, Latency, Accuracy Trends & GPU Alerts", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)

    # 3. Secure Boundary (Bridge)
    b.add_group_box(820, 90, 120, 680, "ZERO-TRUST BRIDGE\n(Firewall / NAT)", bg_color="#fff5f5", stroke_color="#e03131")
    n_zt = b.add_node(830, 280, 100, 220, "Outbound Poll\nOnly (HTTPS/TLS)\n\nZero Inbound\nPorts Open\non DGX\n\n(No Public IP\nNeeded)", bg_color="#ffe3e3", stroke_color="#c92a2a", text_color="#c92a2a", font_size=12)

    # 4. NVIDIA DGX Tier (Right)
    b.add_group_box(980, 90, 540, 680, "ON-PREM NVIDIA DGX COMPUTE CLUSTER", bg_color="#f3f0ff", stroke_color="#7950f2")
    n_dgx_worker = b.add_node(1010, 140, 480, 90, "DGX Worker Supervisor Pool (Celery / Asyncio)\n- Outbound long-polling SQS for CAD jobs\n- Manages isolated per-job workspaces: /tmp/sketchcad_jobs/{id}/\n- Handles timeouts, retries, and artifact uploads", bg_color="#ede9fe", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    n_adk_pipeline = b.add_node(1010, 260, 480, 130, "SketchCAD Google ADK Multi-Agent Core\n1. Analyst Agent: Multimodal vision extraction (DrawingSpec)\n2. Draftsman Agent: 14 geometry math solvers + CADCraft MCP\n3. Critic Agent: 5-point feature audit & accuracy scoring\n4. Refiner Agent: Targeted patch loop (up to 4 iterations)", bg_color="#ffffff", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    n_headless_cad = b.add_node(1010, 420, 230, 110, "Headless CAD Engine\n- Dockerized CADCraft CLI\n- FreeCAD XML-RPC (Xvfb)\n- ezdxf & SVG Generator\n- Evaluator (Watertightness)", bg_color="#ede9fe", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    n_vllm_gpu = b.add_node(1260, 420, 230, 110, "vLLM Inference Cluster\n- Qwen 3.8 27B / Vision\n- Tensor Parallelism (A100s)\n- Continuous Batching\n- `enable_thinking: False`", bg_color="#e6fcf5", stroke_color="#0ca678", text_color="#087f5b", font_size=13)

    n_evaluator = b.add_node(1010, 560, 480, 80, "Geometric Evaluator & Watertightness Audit\nBoundary-representation verification (gap <= 0.05mm, tangency <= 0.1 deg)\nSaves OUT.dxf, OUT.png, OUT.svg, and critic_report.json", bg_color="#ffffff", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    # Connecting Arrows
    b.add_arrow(280, 190, 370, 190, "1. Upload Drawing", stroke_color="#1971c2", start_id=n_user, end_id=n_alb)
    b.add_arrow(550, 340, 550, 370, "2. Store Image", stroke_color="#e8590c", start_id=n_fastapi, end_id=n_s3)
    b.add_arrow(660, 340, 660, 370, "3. Enqueue Job", stroke_color="#e8590c", start_id=n_fastapi, end_id=n_sqs)

    b.add_arrow(1010, 185, 750, 400, "4. DGX Worker Polls SQS (Outbound HTTPS)", stroke_color="#7048e8", start_id=n_dgx_worker, end_id=n_sqs)
    b.add_arrow(1010, 210, 550, 420, "5. Fetch Drawing from S3 Presigned URL", stroke_color="#7048e8", start_id=n_dgx_worker, end_id=n_s3)

    b.add_arrow(1250, 230, 1250, 260, "Execute Agent Pipeline", stroke_color="#7048e8", start_id=n_dgx_worker, end_id=n_adk_pipeline)
    b.add_arrow(1120, 390, 1120, 420, "MCP Tool Calls", stroke_color="#7048e8", start_id=n_adk_pipeline, end_id=n_headless_cad)
    b.add_arrow(1370, 390, 1370, 420, "VLM Vision & Reasoning", stroke_color="#0ca678", start_id=n_adk_pipeline, end_id=n_vllm_gpu)

    b.add_arrow(1120, 530, 1120, 560, "Inspect Geometry", stroke_color="#7048e8", start_id=n_headless_cad, end_id=n_evaluator)
    b.add_arrow(1010, 600, 550, 440, "6. Upload OUT.dxf / SVG / Score to S3", stroke_color="#0ca678", start_id=n_evaluator, end_id=n_s3)
    b.add_arrow(1010, 220, 750, 520, "7. Publish Real-time Logs & Score", stroke_color="#7048e8", start_id=n_dgx_worker, end_id=n_redis)

    b.add_arrow(570, 540, 280, 470, "8. WebSocket Push to Client", stroke_color="#1971c2", start_id=n_redis, end_id=n_ws_client)
    b.add_arrow(370, 420, 280, 330, "9. Download Watertight DXF & SVG", stroke_color="#1864ab", start_id=n_s3, end_id=n_react)

    return b.build_markdown()


def generate_pipeline_diagram():
    b = ExcalidrawSceneBuilder("SketchCAD Agent Execution & Refinement Lifecycle")

    # Header
    b.add_node(250, 20, 850, 50, "SketchCAD Production Agent Execution & Self-Correction Pipeline", bg_color="#5f3dc4", stroke_color="#364fc7", text_color="#ffffff", font_size=18)

    # Stages (Horizontal / Flow)
    # Stage 1: Intake & Preprocessing
    b.add_group_box(40, 90, 230, 680, "STAGE 1: INGESTION", bg_color="#f8f9fa", stroke_color="#adb5bd")
    s1_1 = b.add_node(60, 140, 190, 70, "1. User Input\nPNG / JPEG / PDF\n(Scanned Blueprint)", bg_color="#ffffff", stroke_color="#495057", text_color="#212529", font_size=13)
    s1_2 = b.add_node(60, 250, 190, 100, "2. Preprocessing\n- DPI Normalization\n- Bounding Crop\n- Contrast Auto-Level\n- Metadata Tagging", bg_color="#e9ecef", stroke_color="#495057", text_color="#212529", font_size=13)
    s1_3 = b.add_node(60, 390, 190, 80, "3. Storage & Queue\nS3 Bucket Ingest\nSQS Message Dispatch\nJob State: 'PENDING'", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    s1_4 = b.add_node(60, 510, 190, 80, "4. Worker Claim\nDGX Sandbox Setup\n`/tmp/jobs/{job_id}/`\nJob State: 'ANALYZING'", bg_color="#ede9fe", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    # Stage 2: Perception (Analyst)
    b.add_group_box(300, 90, 260, 680, "STAGE 2: PERCEPTION (ANALYST)", bg_color="#e7f5ff", stroke_color="#1971c2")
    s2_1 = b.add_node(320, 140, 220, 90, "Analyst Agent\n(Qwen 3.8 Vision VLM)\nPrompt Directive: Anchor\nOrigin (0,0) at Bottom-Left", bg_color="#ffffff", stroke_color="#1971c2", text_color="#1864ab", font_size=13)
    s2_2 = b.add_node(320, 270, 220, 120, "Multimodal Dimension\nCallout Extraction\n- Diameters (Ø30, Ø50)\n- Radii (R12, R55)\n- Linear Dimensions (120mm)\n- Angles (60°, 90° chamfers)", bg_color="#d0ebff", stroke_color="#1971c2", text_color="#1864ab", font_size=13)
    s2_3 = b.add_node(320, 430, 220, 110, "Pydantic Validation\n`DrawingSpec` Schema:\n- entities (lines/arcs/circles)\n- relations (tangent/concentric)\n- overall_dimensions\n- part_components", bg_color="#ffffff", stroke_color="#1971c2", text_color="#1864ab", font_size=13)
    s2_4 = b.add_node(320, 580, 220, 60, "Session State Store\n`state['drawing_spec']`\nStream status to Client", bg_color="#d0ebff", stroke_color="#1971c2", text_color="#1864ab", font_size=13)

    # Stage 3: Construction (Draftsman)
    b.add_group_box(590, 90, 270, 680, "STAGE 3: DRAFTING & GEOMETRY", bg_color="#fff4e6", stroke_color="#f76707")
    s3_1 = b.add_node(610, 140, 230, 80, "Draftsman Agent\nReads `DrawingSpec`\nPlans Construction Order\n(Sharp Envelope -> Fillets)", bg_color="#ffffff", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    s3_2 = b.add_node(610, 260, 230, 140, "14 Analytical Geometry Solvers\n(`geometry_tools.py`):\n- `fillet_two_lines`\n- `tangent_lines_two_circles`\n- `circle_line_intersections`\n- `arc_angles` (CCW/CW)\n- `close_joint` (0.05mm)", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    s3_3 = b.add_node(610, 440, 230, 110, "Headless CAD MCP Calls\n- `line(p1, p2)`\n- `arc(center, r, a1, a2)`\n- `circle(center, r)`\n- `save(path='OUT.dxf')`\n- `render(path='OUT.png')`", bg_color="#ffffff", stroke_color="#e8590c", text_color="#d9480f", font_size=13)
    s3_4 = b.add_node(610, 590, 230, 60, "Initial CAD Generated\n`OUT.dxf` & Screenshot\nPass to Refinement Loop", bg_color="#ffe8cc", stroke_color="#e8590c", text_color="#d9480f", font_size=13)

    # Stage 4: Self-Correction Loop (Critic + Refiner)
    b.add_group_box(890, 90, 310, 680, "STAGE 4: SELF-CORRECTION LOOP", bg_color="#f3f0ff", stroke_color="#7950f2")
    s4_1 = b.add_node(910, 140, 270, 110, "Visual Critic Agent\nSide-by-side Inspection:\nInput Drawing vs Rendered `OUT.png`\n5-Point Feature Audit:\n(Base, Bulge, Outer, Protrusion, Cuts)", bg_color="#ffffff", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)
    s4_2 = b.add_node(910, 290, 270, 80, "Accuracy Evaluation\n`CriticReport` Schema:\n- accuracy_score (0.0 - 100.0%)\n- is_accurate (bool)\n- errors & rectification_commands", bg_color="#ede9fe", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)
    s4_decision = b.add_node(930, 410, 230, 60, "Score >= 98% OR\nMax 4 Iterations?", bg_color="#7048e8", stroke_color="#5f3dc4", text_color="#ffffff", font_size=14)
    s4_refiner = b.add_node(910, 510, 270, 110, "CAD Refiner Agent\n- Reads `rectification_commands`\n- Performs targeted CAD edits\n- Recalculates tangent trims\n- Overwrites `OUT.dxf` & renders", bg_color="#ffffff", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)
    s4_loop = b.add_node(910, 660, 270, 50, "LoopAgent Iteration Counter (1..4)", bg_color="#ede9fe", stroke_color="#7048e8", text_color="#5f3dc4", font_size=13)

    # Stage 5: Export & Delivery
    b.add_group_box(1230, 90, 250, 680, "STAGE 5: DELIVERY & METRICS", bg_color="#e6fcf5", stroke_color="#0ca678")
    s5_1 = b.add_node(1250, 140, 210, 90, "`evaluator.py` Metric Gate\n- Watertightness (<= 0.05mm)\n- Tangency angle (<= 0.1 deg)\n- Strict PASS / FAIL validation", bg_color="#ffffff", stroke_color="#0ca678", text_color="#087f5b", font_size=13)
    s5_2 = b.add_node(1250, 270, 210, 90, "Multi-Format Export\n- Final Watertight `OUT.dxf`\n- High-Res Vector `OUT.svg`\n- FreeCAD `.FCStd` / STEP\n- Metadata Audit JSON", bg_color="#c3fae8", stroke_color="#0ca678", text_color="#087f5b", font_size=13)
    s5_3 = b.add_node(1250, 400, 210, 90, "Cloud S3 Storage Sync\nPush artifacts to AWS S3\nMark Job as 'COMPLETED'\nRecord accuracy to Database", bg_color="#ffffff", stroke_color="#0ca678", text_color="#087f5b", font_size=13)
    s5_4 = b.add_node(1250, 530, 210, 110, "Interactive Web Showcase\n- Browser renders 2D CAD\n- Layer & Dimension Toggle\n- Visual diff with original\n- One-click File Download", bg_color="#c3fae8", stroke_color="#0ca678", text_color="#087f5b", font_size=13)

    # Arrows connecting the lifecycle
    b.add_arrow(155, 210, 155, 250, "Raw Raster/PDF", stroke_color="#495057", start_id=s1_1, end_id=s1_2)
    b.add_arrow(155, 350, 155, 390, "Normalized Image", stroke_color="#495057", start_id=s1_2, end_id=s1_3)
    b.add_arrow(155, 470, 155, 510, "Dequeue Task", stroke_color="#7048e8", start_id=s1_3, end_id=s1_4)
    b.add_arrow(250, 550, 320, 185, "Handover to Analyst", stroke_color="#1971c2", start_id=s1_4, end_id=s2_1)

    b.add_arrow(430, 230, 430, 270, "Multi-Pass Vision", stroke_color="#1971c2", start_id=s2_1, end_id=s2_2)
    b.add_arrow(430, 390, 430, 430, "Structured Spec", stroke_color="#1971c2", start_id=s2_2, end_id=s2_3)
    b.add_arrow(430, 540, 430, 580, "Store in State", stroke_color="#1971c2", start_id=s2_3, end_id=s2_4)

    b.add_arrow(540, 610, 610, 180, "Pass DrawingSpec", stroke_color="#e8590c", start_id=s2_4, end_id=s3_1)
    b.add_arrow(725, 220, 725, 260, "Solve Fillets & Arcs", stroke_color="#e8590c", start_id=s3_1, end_id=s3_2)
    b.add_arrow(725, 400, 725, 440, "Exact CAD Coordinates", stroke_color="#e8590c", start_id=s3_2, end_id=s3_3)
    b.add_arrow(725, 550, 725, 590, "Export OUT.dxf", stroke_color="#e8590c", start_id=s3_3, end_id=s3_4)

    b.add_arrow(840, 620, 910, 195, "Enter Refinement Loop", stroke_color="#7048e8", start_id=s3_4, end_id=s4_1)
    b.add_arrow(1045, 250, 1045, 290, "5-Point Feature Audit", stroke_color="#7048e8", start_id=s4_1, end_id=s4_2)
    b.add_arrow(1045, 370, 1045, 410, "Check Threshold", stroke_color="#7048e8", start_id=s4_2, end_id=s4_decision)

    # Decision branches
    b.add_arrow(1045, 470, 1045, 510, "No (<98%) -> Rectify", stroke_color="#c92a2a", start_id=s4_decision, end_id=s4_refiner)
    b.add_arrow(1045, 620, 1045, 660, "Execute Patches", stroke_color="#7048e8", start_id=s4_refiner, end_id=s4_loop)
    b.add_arrow(910, 685, 870, 420, "Re-Inspect Next Iteration", stroke_color="#7048e8", start_id=s4_loop, end_id=s4_1)

    b.add_arrow(1160, 440, 1250, 185, "Yes (>=98%) -> Exit Loop", stroke_color="#0ca678", start_id=s4_decision, end_id=s5_1)
    b.add_arrow(1355, 230, 1355, 270, "Tolerance Confirmed", stroke_color="#0ca678", start_id=s5_1, end_id=s5_2)
    b.add_arrow(1355, 360, 1355, 400, "Package Artifacts", stroke_color="#0ca678", start_id=s5_2, end_id=s5_3)
    b.add_arrow(1355, 490, 1355, 530, "Notify Browser via WS", stroke_color="#0ca678", start_id=s5_3, end_id=s5_4)

    return b.build_markdown()


def main():
    workspace_dir = "/home/udhayan/Agent/my_agent"
    obsidian_vault_dir = "/home/udhayan/Documents/Obsidian Vault"

    print("Generating Diagram 1: Hybrid Cloud-to-DGX Production Architecture...")
    arch_md = generate_architecture_diagram()
    arch_file_local = os.path.join(workspace_dir, "sketchcad_hybrid_cloud_architecture.excalidraw.md")
    with open(arch_file_local, "w", encoding="utf-8") as f:
        f.write(arch_md)
    print(f"Saved: {arch_file_local}")

    print("Generating Diagram 2: Agent Execution & Refinement Lifecycle...")
    pipe_md = generate_pipeline_diagram()
    pipe_file_local = os.path.join(workspace_dir, "sketchcad_agent_pipeline_lifecycle.excalidraw.md")
    with open(pipe_file_local, "w", encoding="utf-8") as f:
        f.write(pipe_md)
    print(f"Saved: {pipe_file_local}")

    # Copy to Obsidian Vault if directory exists
    if os.path.exists(obsidian_vault_dir):
        obsidian_excalidraw_dir = os.path.join(obsidian_vault_dir, "Excalidraw")
        os.makedirs(obsidian_excalidraw_dir, exist_ok=True)

        dest_arch = os.path.join(obsidian_excalidraw_dir, "sketchcad_hybrid_cloud_architecture.excalidraw.md")
        dest_pipe = os.path.join(obsidian_excalidraw_dir, "sketchcad_agent_pipeline_lifecycle.excalidraw.md")

        with open(dest_arch, "w", encoding="utf-8") as f:
            f.write(arch_md)
        with open(dest_pipe, "w", encoding="utf-8") as f:
            f.write(pipe_md)
        print(f"Also copied diagrams to Obsidian Vault:\n - {dest_arch}\n - {dest_pipe}")

    print("Done generating Excalidraw diagrams successfully!")

if __name__ == "__main__":
    main()
