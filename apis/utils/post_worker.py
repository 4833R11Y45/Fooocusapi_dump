"""
Do something after generate
"""
import datetime
import os
from pathlib import Path
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apis.models.base import CurrentTask
from apis.utils.file_utils import change_filename, url_path
from apis.utils import file_utils
from apis.utils.sql_client import GenerateRecord
from apis.utils.web_hook import send_result_to_web_hook
from modules.async_worker import AsyncTask
from modules.config import path_outputs, temp_path

ROOT_DIR = file_utils.SCRIPT_PATH
INPUT_PATH = os.path.join(ROOT_DIR, 'inputs')

engine = create_engine(
    f"sqlite:///{path_outputs}/db.sqlite3",
    connect_args={"check_same_thread": False},
    future=True
)
Session = sessionmaker(bind=engine, autoflush=True)
session = Session()

async def post_worker(task: AsyncTask, started_at: int, target_name: str, ext: str):
    final_enhanced = []
    task_status = "finished"
    
    try:
        # Handle final enhanced images
        if task.save_final_enhanced_image_only:
            for item in task.results:
                if temp_path not in str(item):
                    # Get just the filename and date part from the path
                    clean_path = os.path.basename(str(item))
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    # Construct path relative to path_outputs
                    relative_path = os.path.join(current_date, clean_path)
                    final_enhanced.append(relative_path)
            task.results = final_enhanced

        if task.last_stop in ['stop', 'skip']:
            task_status = task.last_stop

        # Clean paths and verify files
        cleaned_results = []
        for result in task.results:
            # Extract just the relative path components
            clean_path = str(result)
            # Remove any URL prefixes
            clean_path = clean_path.replace('http://127.0.0.1:7866/', '')
            # Remove leading slash
            if clean_path.startswith('/'):
                clean_path = clean_path[1:]
            # Remove any duplicate base paths
            clean_path = clean_path.replace(str(path_outputs), '').lstrip('/')
            
            # Construct full path correctly
            full_path = os.path.join(path_outputs, clean_path)
            
            # Normalize path to remove any '..' or duplicate separators
            full_path = os.path.normpath(full_path)
            
            if os.path.exists(full_path):
                # Store relative path in cleaned_results
                cleaned_results.append(clean_path)
            else:
                print(f"Warning: File not found: {full_path}")

        # Update task results with clean relative paths
        task.results = cleaned_results

        # Update database record
        query = session.query(GenerateRecord).filter(GenerateRecord.task_id == task.task_id).first()
        if query:
            query.start_mills = started_at
            query.finish_mills = int(datetime.datetime.now().timestamp() * 1000)
            query.task_status = task_status
            query.progress = 100
            # Convert relative paths to URLs
            query.result = url_path([os.path.join(path_outputs, path) for path in cleaned_results])
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
