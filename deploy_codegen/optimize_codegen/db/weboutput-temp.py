# def_port_num = 5000
# def_path = "./images" 

##########################################################
from flask import Flask
from flask import request
from flask import Response
from flask import stream_with_context
from flask import render_template
from flask import redirect
from flask import url_for
import time
import cv2
import numpy as np
import threading
from queue import Queue
import imageio as iio
from werkzeug.utils import secure_filename
import os
import pytorch_yolov7 as pyyolo

global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd, clickFile, clickCam
g_fname = ""
g_image_path = def_path 
g_mod = "file" #or "camera"
g_state = "stop" # or "stop"
g_isFitst = True
g_cmd = 0  # 'File' 'Camera'
clickFile = 0
clickCam = 0

class Streamer :
    def __init__(self ):
        self.camera = None
        self.width = 640
        self.height = 360
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.mythr = None
        self.run_flag = True
        self.thread_done = 0
        self.src = ""
        self.y7 = pyyolo.PyTorchRun()

    def thread_for_run(self):
        global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd, clickFile, clickCam
        if self.run_flag is False:
            return
        self.clear()
        self.camera = iio.get_reader(self.src)
        meta = self.camera.get_meta_data()
        (self.width, self.height) = meta['source_size']
        self.thread_done = 0
        while self.run_flag:
            if self.update() < 0:
                break
        self.camera.close()
        self.stop()
        self.run_flag = False
        print('Thread Done')
        self.thread_done = 1
        g_state = "stop"
        return

    def thread_stop(self):
        self.run_flag = False
        cnt = 0
        while cnt < 10:
            if self.thread_done == 1:
                print("Run thread stop")
                return 
            else:
                cnt = cnt + 1
                time.sleep(0.1)
        print("Time Out Run thread")


    def stop(self):
        print("SELF>STOP CALLED")
        if self.camera is not None :
            self.camera.close()
            self.clear()
            self.camera= None
        return
            
    def update(self):
        try:
            frame = self.camera.get_next_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except IndexError:
            print("End of Movie")
            return -1
        else:
            # add detection code here
            # kkkhhhllleeeeeeeee
            output = self.y7.do_oneimage(frame)
            self.Q.put(output)
            # self.Q.put(frame)
            # time.sleep(1/25)
        return 0
                          
    def clear(self):
        with self.Q.mutex:
            self.Q.queue.clear()
            
    def read(self):
        return self.Q.get()

    def blank(self):
        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    def stream_gen(self, src):   
        try : 
            while True :
                frame = streamer.bytescode()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit :
            pass
            # g_state = "stop"
            # streamer.stop()
        return
    
    def bytescode(self):
        if self.camera == None:
            frame = self.blank()
        elif self.camera.closed: 
            frame = self.blank()
        else :
            frame = self.read()
            cv2.rectangle( frame, (0,0), (120,30), (0,0,0), -1)
            fps = 'FPS : ' + str(self.fps())
            cv2.putText  ( frame, fps, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
        return cv2.imencode('.jpg', frame )[1].tobytes()
    
    def fps(self):
        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time
        if self.sec > 0 :
            fps = round(1/(self.sec),1)
        else :
            fps = 1
        return fps
                   
    def __exit__(self) :
        print( '* streamer class exit')
        self.wair_for_done()
        self.camera.close()
        return

#######################################################################################
app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def root():
    print('root')
    global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd, clickFile, clickCam
    clickCam = request.args.get('ToggleCamera', default = 0)
    clickFile = request.args.get('ToggleFile', default = 0)
    print(clickCam, clickFile, g_fname) 
    return render_template('index.html', data=g_fname, error=None) # , form_data = form_data)

@app.route('/index', methods=['POST', 'GET'])
def index():
    print('index')
    global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd, clickFile, clickCam
    clickCam = request.args.get('ToggleCamera', default = 0)
    clickFile = request.args.get('ToggleFile', default = 0)
    print(clickCam, clickFile, g_fname) 
    return render_template('index.html', data=g_fname, error=None) # , form_data = form_data)

@app.route('/video_feed', methods=['POST', 'GET'])
def video_feed():
    global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd, clickFile, clickCam
    print("video_feed", g_fname, g_mod, g_state)
    if g_fname != "":
        streamer.src = "%s/%s" % (g_image_path, g_fname)
        print(streamer.src)
        if g_cmd == 'File':
            clickFile = 1
        elif g_cmd == 'Camera':
            clickCamera = 1

        if clickFile != 0:
            print("file play start")
            clickFile = 0
            g_mod = "file" # "camera"
            if g_state == "stop":
                g_state = "run"
                streamer.stat = True
                streamer.run_flag = True
                streamer.mythr = threading.Thread(target=streamer.thread_for_run, daemon=True, name="DetectionService")
                streamer.mythr.start()
                ret = Response(stream_with_context(streamer.stream_gen(streamer.src)),
                    mimetype='multipart/x-mixed-replace; boundary=frame' )
                return ret
            else: 
                # stop file
                print("Stop play ")
                streamer.thread_stop()
                g_state = "stop"
    if clickCam != 0:
        clickCamera = 0
        print("camreq")
        g_mod = "camera" 
        if g_state == "stop":
            streamer.src = "%s/%s" % (g_image_path, g_fname)
            streamer.stat = True
            # run camera
            print("run camera")
            g_state = "run"
            return render_template('index.html', data=g_fname) # rendering
        else: 
            # stop camera
            print("stop camera")
            g_state = "stop"
    return Response(stream_with_context(streamer.stream_gen(streamer.src)),
                    mimetype='multipart/x-mixed-replace; boundary=frame' )
    # return render_template('index.html', data=g_fname, error=None) 


@app.route('/fileupload', methods=['POST', 'GET'])
def file_upload():
    global g_fname, g_mod, g_state, g_image_path, g_isFirst, g_cmd
    print("FIle upload called")
    print(request.method)
    if request.method == 'GET':
        cmd = request.args.get('ToggleFile', default = 0)
        if cmd == 'File Start/Stop':
            print("FILE")
            g_cmd = 'File'
            return video_feed()
        cmd = request.args.get('ToggleCamera', default = 0)
        if cmd == 'Camera Start/Stop':
            print("Camera")
            g_cmd = 'Camera'
            return video_feed()
    elif request.method == 'POST':
        print('bbb')
        file = request.files['file']
        if file.filename:
            filename = secure_filename(file.filename)
            os.makedirs(g_image_path, exist_ok=True)
            file.save(os.path.join(g_image_path, filename))
            print("File Save Done")
            g_fname = filename
            g_isFirst =False
        # return render_template('index.html', data=g_fname) # , error=None) 
        # return redirect('http://localhost:5000') # , data=g_fname) 
        return redirect(url_for('index')) # , data=g_fname) 

############################################
if __name__ == '__main__':
    print("Detection Service")
    streamer = Streamer()
    app.run(host='localhost', port=def_port_num)
