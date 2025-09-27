from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import google.generativeai as genai
import json
import socket
from typing import Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from PIL import Image
import io
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=3)

# Global client instance
gemini_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - just log that we're starting
    logger.info("Food analyzer starting up")
    
    yield
    
    # Shutdown
    executor.shutdown(wait=True)

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def find_free_port() -> int:
    """Find an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def optimize_image(image_data: bytes, max_size: tuple = (1024, 1024), quality: int = 85) -> bytes:
    """Optimize image for API consumption."""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Convert RGBA to RGB if necessary
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        
        # Resize if larger than max_size
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save optimized image
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Image optimization failed: {e}")
        return image_data

@lru_cache(maxsize=1)
def get_optimized_prompt() -> str:
    """Get the optimized prompt for food analysis."""
    return """Analyze this food image and return ONLY a JSON object with this exact structure:
{
  "dish_name": "string",
  "ingredients": ["string"],
  "nutrition": {
    "calories": {"amount": number, "unit": "kcal"},
    "protein": {"amount": number, "unit": "g"},
    "carbs": {"amount": number, "unit": "g"},
    "fat": {"amount": number, "unit": "g"},
    "fiber": {"amount": number, "unit": "g"}
  }
}
Values per 100g. No markdown, no explanation."""

async def analyze_food_image(image_data: bytes) -> Dict[str, Any]:
    """Analyze food image using Gemini API."""
    # Initialize client if not already done
    global gemini_client
    if gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized")
    
    loop = asyncio.get_event_loop()
    
    # Optimize image in thread pool
    optimized_image = await loop.run_in_executor(
        executor, optimize_image, image_data
    )
    
    # Prepare request
    contents = [
        genai.Content(
            role="user",
            parts=[
                genai.Part.from_text(text=get_optimized_prompt()),
                genai.Part(inline_data=genai.Blob(
                    mime_type='image/jpeg', 
                    data=optimized_image
                ))
            ],
        ),
    ]
    
    # Configure generation with minimal tokens
    generate_content_config = genai.GenerateContentConfig(
        temperature=0.1,  # Lower temperature for more consistent output
        max_output_tokens=500,  # Limit output tokens
        response_mime_type="application/json",  # Request JSON response
    )
    
    # Make API call in thread pool
    response = await loop.run_in_executor(
        executor,
        lambda: gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Use faster model
            contents=contents,
            config=generate_content_config,
        )
    )
    
    return response.text

def validate_response(response_text: str) -> Dict[str, Any]:
    """Validate and parse the API response."""
    try:
        # Clean response text
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        
        # Parse JSON
        data = json.loads(cleaned_text)
        
        # Validate structure
        required_fields = ['dish_name', 'ingredients', 'nutrition']
        if not all(field in data for field in required_fields):
            raise ValueError("Missing required fields")
        
        # Ensure ingredients is a list
        if not isinstance(data['ingredients'], list):
            data['ingredients'] = [data['ingredients']]
        
        # Validate nutrition structure
        if not isinstance(data['nutrition'], dict):
            data['nutrition'] = {}
        
        return data
    except Exception as e:
        logger.error(f"Response validation failed: {e}")
        raise ValueError(f"Invalid response format: {e}")

@app.post('/analyze')
async def analyze(image: UploadFile = File(...)):
    """Analyze uploaded food image."""
    # Validate file
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image selected")
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    if image.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Check file size (10MB limit)
    image_data = await image.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB")
    
    try:
        # Analyze image
        result = await analyze_food_image(image_data)
        
        # Validate and parse response
        json_result = validate_response(result)
        
        return JSONResponse(content=json_result)
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@app.get('/health')
async def health_check():
    """Health check endpoint."""
    # Check if API key is available
    api_key_available = bool(os.environ.get("GEMINI_API_KEY"))
    
    return {
        "status": "healthy" if api_key_available else "degraded", 
        "service": "food-analyzer",
        "api_key_configured": api_key_available
    }

if __name__ == '__main__':
    port = int(os.environ.get('PORT', find_free_port()))
    logger.info(f"Starting server on port {port}")
    
    import uvicorn
    uvicorn.run(
        app, 
        host='0.0.0.0', 
        port=port,
        log_level="info",
        access_log=True
    )
