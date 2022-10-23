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

from detect import detect
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
    # Create uploads directory if not exists
    dir_path = make_dir(dir_path=f'{os.getenv("UPLOADS_PATH")}/model/tmp/')

    # Create outputs directory if not exists
    output_path = make_dir(
        dir_path=f'{os.getenv("MODEL_OUTPUTS")}', use_base_path=False
    )

    contents = await file.read()
    if file.filename.split(".")[1] == "jpg" or "jpeg":
        file.filename = "tmp.jpg"
    elif file.filename.split(".")[1] == "mp4":
        file.filename = "tmp.mp4"
    with open(os.path.join(dir_path, "tmp"), "wb") as fp:
        fp.write(contents)
    # Create file from request bytes
    for file_name in os.listdir(dir_path):
        if file_name.split(".")[1] in images:
            file_from_bytes(file.file, dir_path, "tmp.jpg")
        elif file_name.split(".")[1] in videos:
            file_from_bytes(file.file, dir_path, "tmp.mp4")
        else:
            raise Exception
    # Yolo Config Dict
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLOv5 root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    config = {
        "weights": f"{ROOT}/yolov5s.pt",
        "source": dir_path,
        "device": "cpu",  # or gpu number: 0,1,2,3
        "view_img": False,
        "classes": None,
        "update": False,
        "nosave": False,
        "project": f"{ROOT}/{output_path}",
        "name": "exp",
        "exist_ok": False,
        "dnn": False,
        "data": f"{ROOT}/data/coco128.yaml",
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
    output_path = make_dir(dir_path=f'{os.getenv("MODEL_OUTPUTS")}')
    # Return image with objects as response
    for file_name in os.listdir(output_path):
        if file_name.split(".")[-1] == "mp4":
            response = FileResponse(
                f"{output_path}/{file_name}", media_type="video/mp4"
            )
            response.headers["Content-Disposition"] = "attachment; filename=results.mp4"
            return response
        else:
            return FileResponse(str(output_path / file_name))


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
    # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    # colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting
    uvicorn.run("deploy_server:app", host="0.0.0.0", port=5051, reload=True)

# from pathlib import Path
# from fastapi import FastAPI, UploadFile, File, Response, Header
# from numpy import size
# from pydantic import AnyUrl
# from starlette.responses import StreamingResponse, FileResponse
# from PIL import Image
# import io
# import uvicorn
# import torch
# import base64
# import cv2
# import random
# import aiofiles


# app = FastAPI(title='YOLOv5 inference Server')
# DETECTION_URL = "/inference/"


# def get_yolov5():
#     # model = torch.hub.load('./yolov5', 'custom', path='./model/best.pt', source='local')
#     torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
#     model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache

#     return model


# model = get_yolov5()


# @app.post(
#     DETECTION_URL + 'image',
#     responses={
#         200: {
#             "content": {"image/png": {}},
#         }
#     },
#     response_class=Response,
# )
# async def predict(files: list[UploadFile] = File(...)):
#     for file in files:
#         results = model(Image.open(io.BytesIO(await file.read())), size=640)
#         results.render()

#         for im in results.imgs:
#             buffered = io.BytesIO()
#             im_base64 = Image.fromarray(im)
#             im_base64.save(buffered, format="JPEG")
#             # buffered.seek(0)
#     return Response(content=buffered.getvalue(), media_type="image/png")


# @app.post(DETECTION_URL + 'upload_video')
# async def upload_video(file: UploadFile = File(...)):
#     async with aiofiles.open(Path("./input_video.mp4"), 'wb') as out_file:
#         content = await file.read()  # async read
#         await out_file.write(content)  # async write
#     _handle_inference_video()


# @app.get(DETECTION_URL + 'video')
# async def predict_video():
#     response = FileResponse("./input_video.mp4", media_type='video/mp4')
#     response.headers["Content-Disposition"] = "attachment; filename=video.mp4"
#     return response


# def _handle_inference_video():
#     cap = cv2.VideoCapture('input_video.mp4')
#     # frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     ret_val, img0 = cap.read()
#     while not ret_val:
#         count += 1
#         cap.release()
#         if count == nf:  # last video
#             raise StopIteration
#         path = files[count]
#         new_video(path)
#         ret_val, img0 = cap.read()

#     frame += 1
#     s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '


#     results = model('input_video.mp4', size=320)
#     results.save(save_dir='runs/detect/exp')


# def results_to_json(results, model):
#     ''' Converts yolo model output to json (list of list of dicts)'''
#     return [
#         [
#             {
#                 "class": int(pred[5]),
#                 "class_name": model.model.names[int(pred[5])],
#                 "bbox": [int(x) for x in pred[:4].tolist()],  # convert bbox results to int from float
#                 "confidence": float(pred[4]),
#             }
#             for pred in result
#         ]
#         for result in results.xyxy
#     ]


# def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
#     # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
#     # Plots one bounding box on image 'im' using OpenCV
#     assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
#     tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


# def base64EncodeImage(img):
#     ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
#     _, im_arr = cv2.imencode('.jpg', img)
#     im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')

#     return im_b64


# if __name__ == "__main__":
#     # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
#     # colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)]  # for bbox plotting
#     uvicorn.run("deploy_server:app", host='127.0.0.1', port=7777, reload=True)
