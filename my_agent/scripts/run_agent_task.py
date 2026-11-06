import asyncio
import os
import sys
from pathlib import Path
sys.path.insert(0, '/home/udhayan/Agent/my_agent')
from dotenv import load_dotenv
from google.genai import types
from google.adk.runners import InMemoryRunner

load_dotenv()

async def main():
    img_path = '/home/udhayan/.gemini/antigravity/brain/57f5082b-df03-439c-bcdf-d07b69293d45/.user_uploaded/media_1793864437518_2685728f.png'
    if not os.path.exists(img_path):
        print(f"Error: image {img_path} not found.")
        sys.exit(1)

    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    part_img = types.Part.from_bytes(data=img_bytes, mime_type='image/png')
    prompt_text = "Convert this technical drawing to 2D CAD in FreeCAD. Set h = 100 mm. Extract and draw the complete mechanical profile including the base, side profiles, outer circular arc (R=55 mm), central circular hole (R=30 mm), internal slots/cutouts, and angled tabs. Refine through the loop until accuracy is >= 98%."
    part_txt = types.Part.from_text(text=prompt_text)

    content = types.Content(role='user', parts=[part_img, part_txt])

    from sketchcad_agent.agent import root_agent
    runner = InMemoryRunner(agent=root_agent)
    session_id = 'session_run_user'
    await runner.session_service.create_session(app_name=runner.app_name, user_id='user', session_id=session_id)

    print(f"=== Starting SketchCAD agent execution on {img_path} ===")
    async for event in runner.run_async(user_id='user', session_id=session_id, new_message=content):
        agent_name = getattr(event, 'author', None) or 'system'
        if event.content:
            for part in event.content.parts:
                if part.text:
                    print(f"[{agent_name}]: {part.text.strip()}")
                if hasattr(part, 'function_call') and part.function_call:
                    fc = part.function_call
                    print(f"[{agent_name} TOOL CALL]: {fc.name}({str(fc.args)[:120]})")
                if hasattr(part, 'function_response') and part.function_response:
                    fr = part.function_response
                    print(f"[{agent_name} TOOL RESP]: {fr.name} -> {str(fr.response)[:120]}")

    print("\n=== Execution finished successfully ===")

if __name__ == '__main__':
    asyncio.run(main())
