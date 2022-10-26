import glob
import os
# Add root to sys path
import sys
from os.path import abspath, dirname
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from detect_server import detect
from fastapi.responses import FileResponse
# Scripts
from filesystem import file_from_bytes, make_dir

app = FastAPI()

images = ("bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp")
videos = ("asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv")


# Detect endpoint
@app.post("/detect/")
async def create_files(file: UploadFile = File(...)):
    """
    Receives a list of files and saves them to model paths
    """
    if os.getenv("UPLOADS_PATH") is None or os.getenv("UPLOADS_PATH") != f"{os.getcwd()}/upload":
        os.environ["UPLOADS_PATH"] = f"{os.getcwd()}/upload"
    if os.getenv("MODEL_OUTPUTS") is None or os.getenv("MODEL_OUTPUTS") != f"{os.getcwd()}/output":
        os.environ["MODEL_OUTPUTS"] = f"{os.getcwd()}/output"

    # Create uploads directory if not exists
    dir_path = make_dir(dir_path=f'{os.getenv("UPLOADS_PATH")}/model/tmp/', use_base_path=False)

    # Create outputs directory if not exists
    output_path = make_dir(
        dir_path=f'{os.getenv("MODEL_OUTPUTS")}', use_base_path=False
    )

    contents = await file.read()
    with open(os.path.join(dir_path, file.filename), "wb") as fp:
        fp.write(contents)
    # Yolo Config Dict
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv3 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    config = {
        "weights": f"{ROOT}/yolov3.pt",
        "source": f"{dir_path}/{file.filename}",
        "device": "cpu",  # or gpu number: 0,1,2,3
        "view_img": False,
        "classes": None,
        "update": False,
        "nosave": False,
        "project": f"{output_path}",
        "name": "exp",
        "exist_ok": False,
        "dnn": False,
        "half": False,
        "imgsz": (640, 640),
        "visualize": False,
        "augment": False,
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "agnostic_nms": False,
        "max_det": 1000,
        "save_crop": False,
        "line_thickness": 3,
        "save_txt": False,
        "view_img": False,
        "hide_labels": False,
        "hide_conf": False,
        "save_conf": False,
    }
    # Yolo Detect Objects
    detect(config)

    # Image with objects path
    # output_path = make_dir(dir_path=f'{os.getenv("MODEL_OUTPUTS")}', use_base_path=False)

    # Return image with objects as response
    for file_name in os.listdir(output_path):
        if file_name.split(".")[-1] == "mp4":
            response = FileResponse(
                f"{output_path}/{file_name}", media_type="video/mp4"
            )
            response.headers["Content-Disposition"] = "attachment; filename=results.mp4"
            return response
        else:
            break
    return FileResponse(str(output_path / file.filename))


# Home endpoint, returns simple html
@app.get("/")
async def main():
    content = """
<!-- Font Awesome -->
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.8.2/css/all.css">
<!-- Google Fonts -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Roboto:300,400,500,700&display=swap">
<!-- Bootstrap core CSS -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.5.0/css/bootstrap.min.css" rel="stylesheet">
<!-- Material Design Bootstrap -->
<link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
<body>
<form class="text-center border border-light p-5" action="/detect/" enctype="multipart/form-data" method="post">
<h1>Detecta Objetos en Imágenes</h1>
<p> Esta demo usa yolov5 y fastapi</p>
<div>
  <label for="files" class="btn ">Seleccionar Imagen</label>
  <input name="file" id="files" style="visibility:hidden;" type="file">
  <label for="btnUpload" class="btn btn-primary">Detectar Objetos!</label>
  <input name="btnUpload" id="btnUpload" style="visibility:hidden;" type="submit">
</div>
    """
    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run("deploy_server:app", host="0.0.0.0", port=5051)
