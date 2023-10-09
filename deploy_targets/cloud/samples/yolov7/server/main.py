# Run app: uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}" --reload

import asyncio
import subprocess
from pathlib import Path

import aiofiles

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

MEDIA_DIR = Path("/tmp/yolov7")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

# Mount the static directory at the path "/static"
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class InferenceException(Exception):
    def __init__(self, error_message: str):
        self.error_message = error_message


@app.get("/")
async def read_root():
    with (STATIC_DIR / "index.html").open() as f:
        content = f.read()
    return HTMLResponse(content=content)


async def save_file(temp_dir: Path, file: UploadFile):
    file_path = temp_dir / file.filename
    async with aiofiles.open(file_path, 'wb') as out_file:
        while True:
            chunk = await file.read(8192)
            if not chunk:
                break
            await out_file.write(chunk)
    return file_path


async def run_inference(file_path: Path, file_ext: str):
    cmd = [
        "python",
        "/model/repo/detect.py",
        "--weights", "/model/yolo7-e6e.pt",
        "--conf", "0.25",
        "--img-size", "640",
        "--source", str(file_path),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise InferenceException(stderr.decode())
    return file_path.with_suffix(f'_result.{file_ext}')


@app.post("/image")
async def process_image(file: UploadFile):
    file_path = await save_file(MEDIA_DIR, file)
    try:
        result_path = await run_inference(file_path, "jpg")  # Assuming images are jpg
    except InferenceException as e:
        err_msg = e.error_message if hasattr(e, e.error_message) else repr(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {err_msg}")
    return FileResponse(result_path)


@app.post("/video")
async def process_video(file: UploadFile):
    file_path = await save_file(MEDIA_DIR, file)
    try:
        result_path = await run_inference(file_path, "mp4")  # Assuming videos are mp4
    except InferenceException as e:
        err_msg = e.error_message if hasattr(e, e.error_message) else repr(e)
        raise HTTPException(status_code=500, detail=f"Inference failed: {err_msg}")
    return FileResponse(result_path)

