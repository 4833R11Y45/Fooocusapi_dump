"""
Generate API routes
"""
from fastapi import (
    APIRouter, Depends, Header, Query, UploadFile, HTTPException
)
from sse_starlette.sse import EventSourceResponse
import base64
import os

from apis.models.base import (
    DescribeImageResponse,
    DescribeImageType,
    CurrentTask, GenerateMaskRequest
)
from apis.utils.api_utils import api_key_auth
from apis.utils.call_worker import (
    async_worker,
    stream_output,
    binary_output,
    generate_mask as gm
)
from apis.models.requests import CommonRequest, DescribeImageRequest
from apis.utils.img_utils import read_input_image
from modules.util import HWC3

secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


@secure_router.post(
    path="/v1/engine/generate/",
    summary="Generate endpoint all in one",
    tags=["GenerateV1"])
async def generate_routes(
        common_request: CommonRequest,
        accept: str = Header(None)):
    try:
        # Force image number to 1 for consistent output
        common_request.image_number = 1
        
        # Get the result using async worker
        result = await async_worker(request=common_request, wait_for_result=True)
        
        if result and isinstance(result, dict) and 'result' in result:
            # Get the first image path from results
            url_path = result['result'][0]
            
            # Extract just the filename from the path
            filename = os.path.basename(url_path)
            
            # Get current date
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            # Construct the actual file path using path_outputs from config
            from modules.config import path_outputs
            image_path = os.path.join(path_outputs, current_date, filename)
            
            if not os.path.exists(image_path):
                raise HTTPException(
                    status_code=404,
                    detail=f"Generated image not found at {image_path}"
                )
            
            # Read the image and convert to base64
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Return base64 string in JSON response
            return {"image_base64": f"data:image/png;base64,{img_data}"}
            
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate image"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@secure_router.post(
    path="/v1/tools/generate_mask",
    summary="Generate mask endpoint",
    tags=["GenerateV1"])
async def generate_mask(mask_options: GenerateMaskRequest) -> str:
    """
    Generate mask endpoint
    """
    return await gm(request=mask_options)


@secure_router.post(
    path="/v1/tools/describe-image",
    response_model=DescribeImageResponse,
    tags=["GenerateV1"])
async def describe_image(
        request: DescribeImageRequest):
    """\nDescribe image\n
    Describe image, Get tags from an image
    Arguments:
        request {DescribeImageRequest} -- Describe image request
    Returns:
        DescribeImageResponse -- Describe image response, a string
    """
    image = request.image
    image_type = request.image_type
    if image_type == DescribeImageType.photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        interrogator = default_interrogator_anime
    img = HWC3(await read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)


@secure_router.post("/v1/engine/control", tags=["GenerateV1"])
async def stop_engine(action: str):
    """Stop or skip engine"""
    if action not in ["stop", "skip"]:
        return {"message": "Invalid control action"}
    if CurrentTask.task is None:
        return {"message": "No task running"}
    from ldm_patched.modules import model_management
    ct = CurrentTask.task
    ct.last_stop = action
    if ct.processing:
        model_management.interrupt_current_processing()
    return {"message": f"task {action}ed"}
