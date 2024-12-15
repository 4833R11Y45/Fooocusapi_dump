"""
Calls the worker with the given params.
"""
import asyncio
import io
import os
import json
import uuid
import datetime

from PIL import Image

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Response
import base64

from apis.models.base import CurrentTask, GenerateMaskRequest
from apis.models.response import RecordResponse
from apis.utils.api_utils import params_to_params
from apis.utils.pre_process import pre_worker
from apis.utils.sql_client import GenerateRecord
from apis.utils.post_worker import post_worker
from apis.utils.file_utils import url_path

from apis.utils.img_utils import (
    narray_to_base64img, read_input_image
)
from apis.models.requests import CommonRequest

from extras.inpaint_mask import generate_mask_from_image, SAMOptions
from modules.async_worker import AsyncTask, async_tasks
from modules.config import path_outputs


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, '..', 'inputs')

engine = create_engine(
    f"sqlite:///{path_outputs}/db.sqlite3",
    connect_args={"check_same_thread": False},
    future=True
)
Session = sessionmaker(bind=engine, autoflush=True)
session = Session()


async def execute_in_background(task: AsyncTask, raw_req: CommonRequest, in_queue_mills):
    """
    Executes the request in the background.
    :param task: The task to execute.
    :param raw_req: The raw request.
    :param in_queue_mills: The time the request was enqueued.
    :return: The response.
    """
    finished = False
    started = False
    save_name = raw_req.save_name
    ext = raw_req.output_format
    while not finished:
        await asyncio.sleep(0.2)
        if len(task.yields) > 0:
            if not started:
                started = True
                started_at = int(datetime.datetime.now().timestamp() * 1000)
                CurrentTask.ct = RecordResponse(
                    task_id=task.task_id,
                    req_params=json.loads(raw_req.model_dump_json()),
                    in_queue_mills=in_queue_mills,
                    start_mills=started_at,
                    task_status="running",
                    progress=0
                )
                CurrentTask.task = task
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, _, image = product
                CurrentTask.ct.progress = percentage
                CurrentTask.ct.preview = narray_to_base64img(image)
            if flag == 'finish':
                finished = True
                CurrentTask.task = None
                CurrentTask.ct = None
                return await post_worker(task=task, started_at=started_at, target_name=save_name, ext=ext)


async def process_params(request: CommonRequest):
    """
    Processes the params for the worker.
    :param request: The params to be processed.
    :return: The processed params.
    """
    if request.webhook_url is None or request.webhook_url == "":
        request.webhook_url = os.environ.get("WEBHOOK_URL")
    raw_req, request = await pre_worker(request)
    params = params_to_params(request)
    task_id = uuid.uuid4().hex
    # Create AsyncTask with just args parameter
    task = AsyncTask(args=params)
    # Set task_id after creation
    task.task_id = task_id  
    async_tasks.append(task)
    in_queue_mills = int(datetime.datetime.now().timestamp() * 1000)
    session.add(GenerateRecord(
        task_id=task.task_id,
        req_params=json.loads(raw_req.model_dump_json()),
        webhook_url=raw_req.webhook_url,
        in_queue_mills=in_queue_mills
    ))
    session.commit()

    return task, in_queue_mills, raw_req, task_id


async def stream_output(request: CommonRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    task, in_queue_mills, raw_req, task_id = await process_params(request)

    save_name = raw_req.save_name
    ext = raw_req.output_format
    started = False
    finished = False
    while not finished:
        await asyncio.sleep(0.2)
        if len(task.yields) > 0:
            if not started:
                started = True
                CurrentTask.task = task
                started_at = int(datetime.datetime.now().timestamp() * 1000)
                CurrentTask.ct = RecordResponse(
                    task_id=task.task_id,
                    req_params=json.loads(raw_req.model_dump_json()),
                    in_queue_mills=in_queue_mills,
                    start_mills=started_at,
                    task_status="running",
                    progress=0,
                    result=[]
                )
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, title, image = product
                text = json.dumps({
                    "progress": percentage,
                    "preview": "data:image/png;base64," + narray_to_base64img(image) if narray_to_base64img(image) is not None else narray_to_base64img(image),
                    "message": title,
                    "images": []
                })
                CurrentTask.ct.progress = percentage
                CurrentTask.ct.preview = narray_to_base64img(image)
                yield f"{text}\n"
            if flag == 'finish':
                # await post_worker(task=task, started_at=started_at)
                await asyncio.create_task(post_worker(task=task, started_at=started_at, target_name=save_name, ext=ext))
                text = json.dumps({
                    "progress": 100,
                    "preview": None,
                    "message": "Finished",
                    "images": url_path(task.results)
                })
                yield f"{text}\n"
                finished = True
                CurrentTask.task = None
                CurrentTask.ct = None


async def verify_image_file(file_path, max_attempts=10, delay=0.5):
    """Verify that image file exists and is not empty"""
    for attempt in range(max_attempts):
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 0:
                    with open(file_path, 'rb') as f:
                        header = f.read(8)
                        if len(header) > 0:
                            return True
            except (OSError, IOError):
                pass
        await asyncio.sleep(delay)
    return False

async def post_worker(task: AsyncTask, started_at: int, target_name: str, ext: str):
    final_enhanced = []
    task_status = "finished"
    
    try:
        # Handle final enhanced images
        if task.save_final_enhanced_image_only:
            for item in task.results:
                if temp_path not in str(item):
                    # Remove any URL prefixes and clean path
                    clean_path = str(item).replace('http://127.0.0.1:7866/', '')
                    if clean_path.startswith('/'):
                        clean_path = clean_path[1:]
                    # Ensure path is not duplicated
                    if clean_path.startswith(path_outputs):
                        final_enhanced.append(clean_path)
                    else:
                        final_enhanced.append(os.path.join(path_outputs, clean_path))
            task.results = final_enhanced

        if task.last_stop in ['stop', 'skip']:
            task_status = task.last_stop

        # Clean paths and verify files
        cleaned_results = []
        for result in task.results:
            # Clean path
            clean_path = str(result).replace('http://127.0.0.1:7866/', '')
            if clean_path.startswith('/'):
                clean_path = clean_path[1:]
            
            # Ensure path is not duplicated
            if clean_path.startswith(path_outputs):
                full_path = clean_path
            else:
                full_path = os.path.join(path_outputs, clean_path)

            if os.path.exists(full_path):
                cleaned_results.append(full_path)
            else:
                print(f"Warning: File not found: {full_path}")

        # Update task results with clean paths
        task.results = cleaned_results

        # Update database record
        query = session.query(GenerateRecord).filter(GenerateRecord.task_id == task.task_id).first()
        if query:
            query.start_mills = started_at
            query.finish_mills = int(datetime.datetime.now().timestamp() * 1000)
            query.task_status = task_status
            query.progress = 100
            query.result = url_path(cleaned_results)
            finally_result = str(query)
            session.commit()
            
            if query.webhook_url:
                await send_result_to_web_hook(query.webhook_url, finally_result)
            return finally_result
    except Exception as e:
        print(f"Error in post_worker: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    finally:
        CurrentTask.ct = None

async def binary_output(request: CommonRequest, ext: str):
    try:
        task, in_queue_mills, raw_req, task_id = await process_params(request)
        save_name = raw_req.save_name
        started = False
        finished = False
        
        while not finished:
            await asyncio.sleep(0.2)
            if len(task.yields) > 0:
                if not started:
                    started = True
                    CurrentTask.task = task
                    started_at = int(datetime.datetime.now().timestamp() * 1000)
                    CurrentTask.ct = RecordResponse(
                        task_id=task.task_id,
                        req_params=json.loads(raw_req.model_dump_json()),
                        in_queue_mills=in_queue_mills,
                        start_mills=started_at,
                        task_status="running",
                        progress=0,
                        result=[]
                    )
                flag, product = task.yields.pop(0)
                if flag == 'finish':
                    finished = True
                    CurrentTask.task = None
                    CurrentTask.ct = None
                    await post_worker(task=task, started_at=started_at, target_name=save_name, ext=ext)

        # Wait a bit for file to be written
        await asyncio.sleep(1)

        if task.results and len(task.results) > 0:
            result_path = task.results[0]
            
            # Clean the path
            clean_path = str(result_path)
            # Remove URL prefix if present
            clean_path = clean_path.replace('http://127.0.0.1:7866/', '')
            # Remove leading slash if present
            clean_path = clean_path.lstrip('/')
            # Remove any duplicate base paths
            clean_path = clean_path.replace(str(path_outputs), '').lstrip('/')
            
            # Construct final path correctly
            full_path = os.path.normpath(os.path.join(path_outputs, clean_path))
            
            print(f"Looking for file at: {full_path}")
            
            if os.path.exists(full_path):
                with open(full_path, "rb") as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                return Response(
                    content={"image": img_data},
                    media_type="application/json"
                )
            else:
                return Response(
                    status_code=404,
                    content={"detail": f"Image file not found at {full_path}"}
                )

        else:
            return Response(
                status_code=404,
                content={"detail": "No results found"}
            )

    except Exception as e:
        print(f"Error in binary_output: {str(e)}")
        return Response(
            status_code=500,
            content={"detail": f"Error processing image: {str(e)}"}
        )
    
async def async_worker(request: CommonRequest, wait_for_result: bool = False) -> dict:
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    :param wait_for_result: If True, the function will wait for the result.
    :return: The result of the task.
    """
    task, in_queue_mills, raw_req, task_id = await process_params(request)

    if wait_for_result:
        res = await execute_in_background(task, raw_req, in_queue_mills)
        return json.loads(res)

    asyncio.create_task(execute_in_background(task, raw_req, in_queue_mills))
    return RecordResponse(task_id=task_id, task_status="pending").model_dump()


# This function copy from webui.py
async def generate_mask(request: GenerateMaskRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    :return: The result of the task.
    """
    extras = {}
    sam_options = None
    image = await read_input_image(request.image)
    if request.mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = request.cloth_category
    elif request.mask_model == 'sam':
        sam_options = SAMOptions(
            dino_prompt=request.dino_prompt_text,
            dino_box_threshold=request.box_threshold,
            dino_text_threshold=request.text_threshold,
            dino_erode_or_dilate=request.dino_erode_or_dilate,
            dino_debug=request.dino_debug,
            max_detections=request.sam_max_detections,
            model_type=request.sam_model
        )

    mask, _, _, _ = generate_mask_from_image(image, request.mask_model, extras, sam_options)
    return narray_to_base64img(mask)


async def current_task():
    """
    Returns the current task.
    """
    if CurrentTask.ct is None:
        return []
    return [CurrentTask.ct.model_dump()]
