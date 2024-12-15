from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import uvicorn
from apis.utils.call_worker import async_worker
from apis.models.base import ImagePrompt
from apis.models.requests import CommonRequest, Performance
import base64
from PIL import Image
import io
import os
import asyncio
from datetime import datetime
from modules.config import path_outputs

app = FastAPI()

class ImageInput(BaseModel):
    cn_img: str = Field(..., description="Base64 encoded image")
    cn_type: str = Field(..., description="Type: ImagePrompt, PyraCanny, CPDS, FaceSwap")
    cn_weight: float = Field(..., ge=0.0, le=2.0, description="Weight value between 0-2")
    cn_stop: float = Field(..., ge=0.0, le=1.0, description="Stop value between 0-1")

    @validator('cn_type')
    def validate_cn_type(cls, v):
        allowed_types = ['ImagePrompt', 'PyraCanny', 'CPDS', 'FaceSwap']
        if v not in allowed_types:
            raise ValueError(f'cn_type must be one of {allowed_types}')
        return v

class GenerateRequest(BaseModel):
    images: List[ImageInput] = Field(..., min_items=1, max_items=4)

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        # Create fixed CommonRequest with your specified defaults
        common_request = CommonRequest(
            performance_selection=Performance.QUALITY,
            aspect_ratios_selection="704×1344",
            image_seed=0,
            image_number=1,
            preset="initial",
            prompt="",
            negative_prompt="",
            output_format="png",
            controlnet_image=[
                ImagePrompt(
                    cn_img=img.cn_img,
                    cn_type=img.cn_type,
                    cn_weight=img.cn_weight,
                    cn_stop=img.cn_stop
                ) for img in request.images
            ]
        )

        # Process the request
        result = await async_worker(request=common_request, wait_for_result=True)
        
        if result and isinstance(result, dict) and 'result' in result:
            # Get the first image path from results
            image_path = result['result'][0]
            if isinstance(image_path, str):
                # Add retry mechanism for file access
                max_retries = 3
                retry_delay = 1
                
                for attempt in range(max_retries):
                    try:
                        # Use the image_path directly since it already contains the full path
                        if os.path.exists(image_path):
                            with Image.open(image_path) as img:
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                img_str = base64.b64encode(buffered.getvalue()).decode()
                            return {"image_base64": f"data:image/png;base64,{img_str}"}
                        else:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                            else:
                                raise HTTPException(
                                    status_code=404,
                                    detail=f"Generated image not found at {image_path}"
                                )
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Error processing image: {str(e)}"
                            )
                        await asyncio.sleep(retry_delay)
        
        raise HTTPException(
            status_code=500,
            detail="Failed to generate image or invalid result format"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in generate endpoint: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7865)
