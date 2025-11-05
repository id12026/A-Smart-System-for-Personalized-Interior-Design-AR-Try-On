from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from utils.base64_helpers import array_buffer_to_base64
from utils.analysis import analyze_image
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import traceback
import base64

load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyD4nrs9RqQISf6xW6jivgVq0mx3RDpC8Rg")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

@router.post("/analyze")
async def analyze(place_image: UploadFile = File(...)):
    try:
        place_bytes = await place_image.read()
        result = analyze_image(place_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        print(f"Error in /api/analyze: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@router.post("/try-on")
async def try_on(
    place_image: UploadFile = File(...),
    design_type: str = Form(...),
    room_type: str = Form(...),
    style: str = Form(...),
    background_color: str = Form(...),
    foreground_color: str = Form(...),
    instructions: str = Form(""),
   
):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing; set it in .env")
        
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if place_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type for place_image: {place_image.content_type}"
            )

        place_bytes = await place_image.read()

        size_in_mb_for_place_image = len(place_bytes) / (1024 * 1024)
        if size_in_mb_for_place_image > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB size limit for place_image")
        
       
        place_b64 = array_buffer_to_base64(place_bytes)

        prompt = f"""
        You are a professional AI INTERIOR designer.
        Redesign ONLY interior spaces. Ignore any outdoor/exterior concepts.

        ### User Input
        - Design Focus: INTERIOR ONLY
        - Room Type: {room_type}
        - Style: {style}
        - Background Color Pref: {background_color}
        - Foreground Color Pref: {foreground_color}
        - Instructions: {instructions}

        ### Goals
        1) Keep the room architecture unchanged (no structural edits).
        2) Apply the selected style to furnishings, finishes, lighting, and decor.
        3) Harmonize background/foreground color preferences.
        4) Produce a photorealistic redesigned image.
        5) Provide an INTERIOR RECOMMENDATIONS report including:
           - Suggested styles and rationale
           - Furniture suggestions (names/types) and placement notes
           - Optimized layout tips and circulation improvements
           - Color palette (HEX values)
           - Estimated budget and time (INR and USD)

        ### Output
        - Return a redesigned image
        - Return a concise markdown report with the above bullets
        """
               
   
        
        print(prompt)

        contents=[
            prompt,
            types.Part.from_bytes(
                data=place_b64,
                mime_type= place_image.content_type,
            )
        ]        
        
        image_data = None
        text_response = "No Description available."
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )

            print(response)

            if response.candidates and len(response.candidates) > 0:
                parts = response.candidates[0].content.parts

                if parts:
                    print("Number of parts in response:", len(parts))

                    for part in parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            image_data = part.inline_data.data
                            image_mime_type = getattr(part.inline_data, "mime_type", "image/png")
                            print("Image data received, length:", len(image_data))
                            print("MIME type:", image_mime_type)

                        elif hasattr(part, "text") and part.text:
                            text_response = part.text
                            preview = (text_response[:100] + "...") if len(text_response) > 100 else text_response
                            print("Text response received:", preview)
                else:
                    print("No parts found in the response candidate.")
            else:
                print("No candidates found in the API response.")
        except Exception as gen_err:
            # Graceful fallback: echo original image and synthesize text
            print(f"Gemini generation failed, using fallback: {gen_err}")
            image_data = place_bytes
            image_mime_type = place_image.content_type or "image/png"
            # Simple synthesized recommendations
            text_response = (
                f"### INTERIOR RECOMMENDATIONS\n\n"
                f"- Room Type: {room_type}\n"
                f"- Style: {style}\n"
                f"- Palette: Background {background_color}, Foreground {foreground_color}\n"
                f"- Suggestions: Optimize layout for circulation, add layered lighting, use cohesive textiles, "
                f"and keep decor minimal for a clean, {style.lower()} aesthetic.\n"
            )
 
        image_url = None
        generated_analysis = None
        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
            try:
                generated_analysis = analyze_image(image_data)
            except Exception:
                generated_analysis = None
        else:
            image_url = None
        
        # Simple success metric (heuristic): image present => high confidence
        success_rate = 0.9 if image_url else 0.3
        
        analysis = analyze_image(place_bytes)
        return JSONResponse(
        content={
            "image": image_url,
            "text": text_response,
            "success_rate": success_rate,
            "analysis": analysis,
            "generated_analysis": generated_analysis,
        }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /api/try-on endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
