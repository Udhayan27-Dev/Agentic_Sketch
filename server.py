"""
CAD Inspector Agent Server (DGX-Spark Port 8000)
Serves Autodesk CAD Model Space UI and Agent Inspection API
"""

import os
import json
import base64
from aiohttp import web
from agent.cad_agent import CADInspectorAgent

agent = CADInspectorAgent()

# Helper CORS middleware
@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


async def index_handler(request):
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return web.Response(text=f.read(), content_type="text/html")
    return web.Response(text="<h1>CAD Inspector Agent Server Running</h1>", content_type="text/html")


async def health_handler(request):
    return web.json_response({
        "status": "online",
        "service": "CAD Inspector Agent",
        "accelerator": "NVIDIA GB10 (DGX Spark)",
        "memory_tier": "48GB Unified"
    })


async def sample_handler(request):
    name = request.match_info.get("name", "")
    sample_path = os.path.join(os.path.dirname(__file__), f"{name}.png")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), "bracket_draft.png")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(__file__), "autodesk_primitive_preview.png")
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            return web.Response(body=f.read(), content_type="image/png")
    return web.Response(status=404, text="Sample not found")


async def process_handler(request):
    """Fallback / standard process endpoint for background removal."""
    try:
        data = await request.json()
        b64_str = data.get("image_base64", "")
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_bytes = base64.b64decode(b64_str)

        result = agent.inspect_and_isolate(image_bytes)
        return web.json_response({
            "image_base64": result["isolated_geometry_base64"],
            "processing_time_ms": result["total_execution_ms"],
            "metrics": result["metrics"]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


async def inspect_handler(request):
    try:
        image_bytes = None

        if request.content_type.startswith("application/json"):
            data = await request.json()
            b64_str = data.get("image_base64", "")
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            image_bytes = base64.b64decode(b64_str)

        elif request.content_type.startswith("multipart/form-data"):
            reader = await request.multipart()
            while True:
                field = await reader.next()
                if field is None:
                    break
                if field.name == "image":
                    image_bytes = await field.read()
                    break

        if not image_bytes:
            return web.json_response({"error": "No image payload provided"}, status=400)

        # Execute CAD Inspector Agent
        result = agent.inspect_and_isolate(image_bytes)
        return web.json_response(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)


def create_app():
    app = web.Application(middlewares=[cors_middleware], client_max_size=50 * 1024 * 1024)
    app.router.add_get("/", index_handler)
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/api/sample/{name}", sample_handler)
    app.router.add_post("/api/process", process_handler)
    app.router.add_post("/api/agent/inspect", inspect_handler)
    app.router.add_static("/static/", os.path.join(os.path.dirname(__file__), "static"))
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print("------------------------------------------------------------------")
    print(f"CAD Inspector Agent Server starting on http://0.0.0.0:{port}")
    print("------------------------------------------------------------------")
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=port)
