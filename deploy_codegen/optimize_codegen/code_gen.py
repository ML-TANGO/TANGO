"""
copyright notice
"""

"""
code_gen.py
This module generates template code for RKNN, PyTorch, and ArmNN based on AutoNN's output.
"""
# khlee To Do List
'''
deployment.yaml user editing -> user can be add papy app libs 
tflite code gen.
input source mp4 + jpg in same folder
'''

import os
import socket
import shutil
import zipfile
import numpy as np
import onnx
import tvm
import tvm.relay as relay
import sys
import logging
import threading
import time
import yaml
# for web service
from http.server import HTTPServer, SimpleHTTPRequestHandler
import requests
# import      subprocess
# import      importlib
# code_gen.py
import      torch
import      torch.onnx
import tensorflow as tf
# from        torchvision import  models
# from        onnxruntime.quantization import  quantize_dynamic
# from        onnxruntime.quantization import  QuantType

logging.basicConfig(level=logging.DEBUG, format="(%(threadName)s) %(message)s")


# for docker and project manager
def_top_folder = "/tango/common"    # for docker
def_top_data_folder = "/tango/datasets"    # for docker
def_code_folder_name = "nn_model"

# for TensorRT
def_trt_converter_file_name = "tensorrt-converter.py"
def_trt_inference_file_name = "tensorrt-infer-template.py"
def_trt_myutil_file_name = "./db/myutil.py"
def_trt_calib_cache = "./db/calibration.cache"
def_trt_engine = "v7-16.trt"
def_trt_precision = "fp16" # "int8"
def_trt_conf_thres = 0.4
def_trt_iou_thres = 0.5
def_trt_max_detection = 100

# for TVM
def_TVM_dev_type = 0   # 0 llvm ,1 cuda,  
def_TVM_width = 640 
def_TVM_height = 640 
def_TVM_data_type = "float32"
def_TVM_mod = "yolov9-tvm.model"
def_TVM_param = "yolov9-tvm.param"
def_TVM_myutil_file_name = "./db/myutil.py"


# defualt values
def_nninfo_file = "neural_net_info.yaml"
def_sysinfo_file = "project_info.yaml"
def_deployment_file = "deployment.yaml"

def_task_type = 'detection'  # classification
def_memory_size = 1
def_cpu_type = ""  # 'x86'  # arm
def_acc_type = ""  # 'cpu'  # cpu/cuda/opencl
def_os_type = ""  # ubuntu'  # linux/windows
def_engine_type = ""  # pytorch'  # acl/tvm/tensorrt
def_libs = [] # ["python==3.9", "torch>=1.1.0"]
def_apt = [] # ["vim", "python3.9"]
def_papi = [] # ["flask==1.2.3", "torch>=1.1.0"]

def_deploy_type = ""  # 'cloud'
def_deploy_work_dir = '.'
def_deploy_python_file = "output.py"
def_deploy_entrypoint = ""  # ["run.sh", "-p", "opt1", "arg"]
def_deploy_network_hostip = ""  # '1.2.3.4'
def_deploy_network_hostport = ""  # '8088'
def_deploy_nfs_ip = '1.2.3.4'
def_deploy_nfs_path = "/tango/common/model" 

def_class_file = ""  # 'input.py'
def_model_definition_file = ""
def_weight_pt_file = ""  # 'input.pt'
def_weight_onnx_file = ""  # ' input.onnx'
def_dataset_path = "/tango/datasets"  
def_annotation_file = "dataset.yaml"  # 'coco.dat'

def_newline = '\n'
def_4blank = "    "

def_codegen_port = 8888

def_n2_manual = './db/odroid-n2-manual.txt'
def_m1_manual = './db/odroid-m1-manual.txt'
def_tensorrt_manual = './db/odroid-m1-manual.txt'
def_tvm_manual = './db/odroid-m1-manual.txt'



# for android
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def tf2tflite(input_size = 640, pb_file="model_float32.pb", output_file="mymodel.tflite"):
    input_arrays = ['inputs']
    output_arrays = ['Identity', 'Identity_1', 'Identity_2']
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_file, input_arrays, output_arrays)
    converter.experimental_new_quantizer = False
    # converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(output_file, 'wb') as w:
        w.write(tflite_model)
    return

def onnx2tflite(onnx_filename = "./yolov7-tiny.onnx"):
    # onnx2openvino
    mycmd = "python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py "
    mycmd = mycmd + " --input_model %s" % onnx_filename
    mycmd = mycmd + " --input_shape [1,3,640,640] "
    mycmd = mycmd + " --output_dir .  "
    mycmd = mycmd + " --data_type FP32 "
    mycmd = mycmd + " --output Conv_134,Conv_149,Conv_164"
    os.system(mycmd)

    # openvino2tflite
    os.system("openvino2tensorflow --model_path ./yolov7-tiny.xml  --model_output_path .  --output_no_quant_float32_tflite")

    #move the tflite file to the place
    os.system("cp model_float32.tflite ./tflite_yolov7_test/app/src/main/assets/yolov7-tiny_fp32_640.tflite")
    os.system("rm -f ./tflite_yolov7_test/app/build/intermediates/assets/debug/yolov7-tiny_fp32_640.tflite")

    #apk build
    os.system("./tflite_yolov7_test/gradlew init < /app/enter.txt  && cd ./tflite_yolov7_test && ./gradlew assembleDebug")
    os.system("cp ./tflite_yolov7_test/app/build/outputs/apk/debug/app-debug.apk /app")

    # remove temp file & dirs
    os.system("sudo /bin/rm -rf yolov7-tiny.xml tflite *.onnx *.mapping *.bin *.ptl *.tflite")
    ret = "app-debug.apk"
    return ret



####################################################################
# class for code generation
####################################################################
class CodeGen:
    """Class Definition for Code Generation"""
    # for docker & project manager
    m_current_file_path = def_top_folder
    m_current_userid = ""
    m_current_projectid = ""
    m_current_code_folder = "."

    # for system information
    m_nninfo_file = def_nninfo_file
    m_sysinfo_file = def_sysinfo_file

    # for build
    m_sysinfo_task_type = def_task_type
    m_sysinfo_memory = def_memory_size
    m_sysinfo_cpu_type = def_cpu_type
    m_sysinfo_acc_type = def_acc_type
    m_sysinfo_os_type = def_os_type
    m_sysinfo_engine_type = def_engine_type
    m_sysinfo_libs = def_libs
    m_sysinfo_apt = def_apt
    m_sysinfo_papi = def_papi

    # for deploy
    m_deploy_type = def_deploy_type
    m_deploy_work_dir = def_deploy_work_dir
    m_deploy_entrypoint = def_deploy_entrypoint
    m_deploy_network_hostip = def_deploy_network_hostip
    m_deploy_network_hostport = def_deploy_network_hostport
    m_deploy_nfs_ip = def_deploy_nfs_ip
    m_deploy_nfs_path = def_deploy_nfs_path

    m_sysinfo_lightweight_level = 5
    m_sysinfo_precision_level = 5
    m_sysinfo_preprocessing_lib = ""
    m_sysinfo_vision_lib = ""
    # url, file/directory path, camera device ID number(0-9)
    m_sysinfo_input_method = "./images"  
    # 0=screen, 1=text, url,  directory path #modified
    m_sysinfo_output_method = "0" 
    m_sysinfo_confidence_thresh = 0.7
    m_sysinfo_iou_thresh = 0.5
    m_sysinfo_user_editing = ""  # no/yes
    m_sysinfo_shared_folder = "/tmp"  # shared folder with host

    # for neural network dependent information
    m_nninfo_class_name = ""
    m_nninfo_class_file = def_class_file
    m_nninfo_model_definition_file = def_model_definition_file
    m_nninfo_weight_pt_file = def_weight_pt_file
    m_nninfo_weight_onnx_file = def_weight_onnx_file
    m_nninfo_annotation_file = def_annotation_file
    m_nninfo_labelmap_info = []
    m_nninfo_number_of_labels = 0
    m_nninfo_anchors = []
    m_nninfo_mask = []
    m_nninfo_output_number = 0
    m_nninfo_output_size = []
    m_nninfo_postproc_conf_thres = 0
    m_nninfo_postproc_iou_thres = 0
    m_nninfo_postproc_need_nms = False

    m_nninfo_input_tensor_shape = []
    m_nninfo_input_data_type = ""

    m_converted_file = ""
    m_deploy_python_file = def_deploy_python_file
    m_deployment_file = def_deployment_file
    m_requirement_file = 'requirements.txt'

    m_deploy_network_serviceport = 0

    m_last_run_state = 0
    
    
    ev = threading.Event()
    lock = threading.Lock()
    the_cnt = 0
    thread_run_flag = 1
    thread_done = 0

    m_atwork = 0   # set 1 when run() fuction is called, reset when run() function is finished


    ####################################################################
    def thread_for_run(self):
        while self.thread_run_flag > 0:
            if self.the_cnt > 0:
                logging.debug("Call Run()")
                if self.thread_run_flag > 0:
                    self.m_atwork = 1
                    self.run()
                    self.m_atwork = 0
                    # send status_report
                    self.response()
                    logging.debug("code_gen: send_status_report to manager")
                    with self.lock:
                        self.the_cnt = self.the_cnt - 1
            else:
                event = self.ev.wait(20)
                if event:
                    self.ev.clear()
                    logging.debug("Recv Event")
        self.thread_done = 1
        logging.debug("Thread Done")
        return

    ####################################################################
    def wait_for_done(self):
        self.thread_run_flag = 0
        with self.lock:
            self.the_cnt = 0
        self.ev.set()
        for j in range(3):
            if self.thread_done != 0:
                break
            else:
                time.sleep(1)
        logging.debug("code_gen Module End")
        return

    ####################################################################
    def move_subfolderfile(self, src, dst):
        """
        make a new dst folder and move files to it

        Args:
            src : src file path  not folder name (string)
            dst : target folder path not file name(string)
        Returns: int
            0 : success
            -1 : file error
        """
        tmp_path = self.get_real_filepath(".")
        if dst != "" and dst != ".":
            new_folders = dst.split('/')
            for item in new_folders:
                tmp_path = "%s%s%s" % (tmp_path, "/", item)
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)

        tmp_path = self.get_real_filepath(dst)
        path_folders = src.split('/')
        items = len(path_folders)
        # folder in folder
        for i in range(items):
            if i == (items - 1):
                if os.path.exists(self.get_real_filepath(src)):
                    if not os.path.exists(tmp_path):
                        shutil.copy(self.get_real_filepath(src), tmp_path)
                break
            else:
                # get real path and add path_folders[i]
                # check directory is exist
                tmp_path = "%s%s%s" % (tmp_path, "/", path_folders[i])
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
        return 0            
    

    ####################################################################
    def copy_subfolderfile(self, src, dst, base_dir = ""):
        """
        copy a file of a subfoler to dst folder 

        Args:
            src : src file path  not folder name (string)
            dst : target folder path not file name(string)
        Returns: int
            0 : success
            -1 : file error
        """
        # check if src is a file name or a file path
        # if file path, 
        #     check path already exists or not
        #     if a path does not exist, make the folder
        tmp_path = dst
        if base_dir != "":
            new_folders = base_dir.split('/')
            for item in new_folders:
                tmp_path = "%s%s%s" % (tmp_path, "/", item)
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)

        path_folders = src.split('/')
        items = len(path_folders)
        # folder in folder
        for i in range(items):
            if i == (items - 1):
                if os.path.exists(self.get_real_filepath(src)):
                    res = shutil.copy(self.get_real_filepath(src), tmp_path)
                break
            else:
                # get real path and add path_folders[i]
                # check directory is exist
                tmp_path = "%s%s%s" % (tmp_path, "/", path_folders[i])
                if not os.path.exists(tmp_path):
                    os.makedirs(tmp_path)
        return 0
                    
    ####################################################################
    def set_folder(self, uid, pid):
        """
        Set user id, project id and file path

        Args:
            uid : user id(string)
            pid : project id (string)
        Returns: int
            0 : success
            -1 : file error
        """
        self.m_current_userid = uid
        self.m_current_projectid = pid
        self.m_current_file_path = "%s%s%s%s%s%s" % (def_top_folder, "/", uid, "/", pid, "/")
        self.m_current_code_folder = "%s%s" % (self.m_current_file_path, def_code_folder_name)
        return 0

    ####################################################################
    def get_real_filepath(self, filename):
        """
        Get absolute file path

        Args:
            filename : filename
        Returns: int
            full file path (string)
        """
        ret = "%s%s" % (self.m_current_file_path, filename)
        return ret

    ####################################################################
    def get_code_filepath(self, filename):
        """
         Get absolute file path for code generation

         Args:
             filename : filename
        Returns: int
            full file path (string)
        """
        # if no m_current_code_folder, then make the folder
        try:
            if not os.path.exists(self.m_current_code_folder):
                os.makedirs(self.m_current_code_folder)
        except OSError:
            logging.debug('Error: Creating directory ' + self.m_current_code_folder)
            print('Error: Creating directory ' + self.m_current_code_folder)

        ret = "%s%s%s" % (self.m_current_code_folder, "/", filename)
        return ret

    ####################################################################
    def parse_sysinfo_file(self):
        """
        Read System Information file

        Args: None
        Returns: int
            0 : success
            -1 : file error
        """
        try:
            f = open(self.get_real_filepath(self.m_sysinfo_file), encoding='UTF-8')
        except IOError as err:
            logging.debug("Sysinfo file open error!!")
            return -1

        m_sysinfo = yaml.load(f, Loader=yaml.FullLoader)
        for key, value in sorted(m_sysinfo.items()):
            if key == 'task_type':
                self.m_sysinfo_task_type = value
            elif key == 'memory':
                self.m_sysinfo_memory = int(value)
            elif key == 'target_info':
                lower = value.lower()
                self.m_deploy_type = lower
            elif key == 'cpu':
                self.m_sysinfo_cpu_type = value
            elif key == 'acc':
                self.m_sysinfo_acc_type = value
            elif key == 'os':
                self.m_sysinfo_os_type = value
            elif key == 'engine':
                self.m_sysinfo_engine_type = value
            elif key == 'target_hostip':
                self.m_deploy_network_hostip = value
            elif key == 'target_hostport':
                self.m_deploy_network_hostport = value
            elif key == 'target_serviceport':
                self.m_deploy_network_serviceport = value
            elif key == 'nfs_ip':
                self.m_deploy_nfs_ip = value
            elif key == 'nfs_path':
                self.m_deploy_nfs_path = value
            elif key == 'lightweight_level':
                self.m_sysinfo_lightweight_level = int(value)
            elif key == 'precision_level':
                self.m_sysinfo_precision_level = int(value)
            elif key == 'input_source':
                if type(value) != int:
                    self.m_sysinfo_input_method = value  # url, path, camera ID 
                else:
                    self.m_sysinfo_input_method = value  # url, path, camera ID 
            elif key == 'output_method':
                # 0:graphic, 1:text, path, url
                if type(value) != int:
                    self.m_sysinfo_output_method = "'" + value + "'"  
                else:
                    self.m_sysinfo_output_method = value  
            elif key == 'user_editing':
                self.m_sysinfo_user_editing = value  # yes/no
            elif key == 'confidence_thresh':
                self.m_sysinfo_confidence_thresh = value  
            elif key == 'iou_thresh':
                self.m_sysinfo_iou_thresh = value  
            # elif key == 'dataset':
                # myset = "%s/dataset.yaml" % (value)
                # self.m_nninfo_annotation_file = myset
        f.close()
        return 0

    ####################################################################
    def parse_nninfo_file(self):
        """
        Read Neural Network Information file generated by autonn

        Args: None
        Returns: int
            0 : success
            -1 : file error
        """
        try:
            f = open(self.get_real_filepath(self.m_nninfo_file), encoding='UTF-8')
        except IOError as err:
            logging.debug("nninfo file open error!!")
            return -1
        m_nninfo = yaml.load(f, Loader=yaml.FullLoader)

        self.m_nninfo_number_of_labels = 0
        self.m_nninfo_labelmap_info = None 
        # parse nninfo
        for key, value in sorted(m_nninfo.items()):
            if key == 'weight_file':
                if isinstance(value, list):
                    cnt = len(value)
                    for i in range(cnt):
                        val = value[i]
                        if ".onnx" in val: 
                            self.m_nninfo_weight_onnx_file = val  # .onnx
                        else:
                            self.m_nninfo_weight_pt_file = val
                else:
                    if ".onnx" in value:
                        self.m_nninfo_weight_onnx_file = value  # .onnx
                    else:
                        self.m_nninfo_weight_pt_file = value
            elif key == 'nc':
                self.m_nninfo_number_of_labels = int(value)
            elif key == 'names':
                self.m_nninfo_labelmap_info = value
            elif key == 'input_tensor_shape':
                if isinstance(value, list):
                    cnt = len(value)
                    tmp1 = list(range(0, cnt))
                    for i in range(cnt):
                        tmp1[i] = int(value[i])
                    self.m_nninfo_input_tensor_shape = tmp1
            elif key == 'input_data_type': # fp16
                self.m_nninfo_input_data_type = value
            elif key == 'anchors': # None
                self.m_nninfo_anchors = value
            elif key == 'output_number': # 3
                self.m_nninfo_output_number = int(value)
            elif key == 'output_size': # [[1, 84, 8400], [1, 84, 8400]] # first one for traing, last one for inf.
                self.m_nninfo_output_size = value
            elif key == 'stride' : # [8, 16, 32]
                self.m_nninfo_stride = value # new value
            elif key == 'nms': # True
                self.m_nninfo_postproc_need_nms = value
            elif key == 'conf_thres': #0.25
                self.m_nninfo_postproc_conf_thres = float(value)
            elif key == 'iou_thres': #0.45
                self.m_nninfo_postproc_iou_thres = float(value)
        if self.m_nninfo_number_of_labels != 0 and self.m_nninfo_labelmap_info != None: 
            annotation_file = "./%s" % (self.m_nninfo_annotation_file)
            ordict = {}
            for i in range(self.m_nninfo_number_of_labels):
                ordict[i] = self.m_nninfo_labelmap_info[i] 
            nc_head = {'nc':self.m_nninfo_number_of_labels}
            try:
                f = open(annotation_file, "w")
            except IOError as err:
                logging.debug("annotation file open error!!")
                return -1
            yaml.dump(nc_head, f)
            d = {'names':ordict}
            yaml.dump(d, f)
            f.close()
        return 0

    ####################################################################
    def run(self):
        """
        Generate Template code

        Args: None
        Returns: None
        """
        self.m_last_run_state = 0
        # m_nninfo_file = def_nninfo_file
        # m_sysinfo_file = def_sysinfo_file
        self.m_sysinfo_libs = def_libs
        self.m_sysinfo_apt = def_apt
        self.m_sysinfo_papi = def_papi
        self.m_nninfo_user_libs = []
        self.m_nninfo_weight_onnx_file = ""
        self.parse_nninfo_file()
        self.parse_sysinfo_file()

        # if there are files in the target folder, remove them
        self.clear()

        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)

        if self.m_sysinfo_task_type == "classification":
            if self.m_sysinfo_engine_type == 'pytorch':
                self.m_sysinfo_libs = []
                self.m_sysinfo_apt = ['vim', 'python3.9']
                self.m_sysinfo_papi = ['torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'albumentations']
                self.m_deploy_entrypoint = self.m_deploy_python_file
                if not os.path.exists(self.m_current_code_folder):
                    os.makedirs(self.m_current_code_folder)
                # make nn_model folder
                if not os.path.exists(self.m_current_code_folder):
                    os.makedirs(self.m_current_code_folder)
                # copy pt file
                pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
                shutil.copy(pt_path, self.m_current_code_folder)
                # copy template code
                codefile_path = "%s%s%s" % (self.m_current_code_folder, "/", self.m_deploy_python_file)
                f = open(codefile_path, "w")
                # copy heading 
                f.write("#!/usr/bin/python\n")
                f.write("# -*- coding: utf-8 -*-\n")
                f.write("DEF_IMG_PATH = %s\n" % self.m_sysinfo_input_method) 
                f.write("DEF_ACC = %s\n" % "\"cpu\"") # only for testing self.m_sysinfo_acc_type) 
                f.write("DEF_PT_FILE = \"%s\"\n\n\n" % self.m_nninfo_weight_pt_file)
                try:
                    f1 = open("./db/resnet152.db", 'r')
                except IOError as err:
                    logging.debug("resnet162 db open error")
                    self.m_last_run_state = 0
                    return -1
                for line1 in f1:
                    f.write(line1)
                f1.close()
                f.close()
                self.make_deployment_yaml()  
            else:
                print("the inference engine is not support for classification")
            self.m_last_run_state = 0
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            return 0

        # tflite for galaxy 
        if self.m_sysinfo_engine_type == 'tflite':   
            if not os.path.exists(self.m_current_code_folder):
                os.makedirs(self.m_current_code_folder)
            # read  onnx file
            my_onnx = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_onnx_file)
            # onnx2tflitefile
            ret = onnx2tflite(onnx_filename = my_onnx)
            # copy apk to nn_model folder
            shutil.copy(ret, self.m_current_code_folder)
            self.m_last_run_state = 0
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            return 0

        # pytorch
        if self.m_sysinfo_engine_type == 'pytorch':
            self.gen_pytorch_code() # web or not
            self.make_deployment_yaml()  
            self.m_last_run_state = 0
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            return 0
        # acl
        elif self.m_sysinfo_engine_type == 'acl':
            self.m_deploy_entrypoint = [self.m_deploy_python_file]
            self.m_sysinfo_libs = ['mali-fbdev']
            self.m_sysinfo_apt = ['clinfo', 'ocl-icd-libopnecl1',
                                  'ocl-icd-opencl-dev', 'python3-opencv', 'python3-pip']
            self.m_sysinfo_papi = []
            self.gen_acl_code()
            self.make_deployment_yaml()  
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            self.m_last_run_state = 0

        elif self.m_sysinfo_engine_type == "tensorrt":
            self.gen_trt_code()
            self.make_deployment_yaml()  
            self.m_last_run_state = 0
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            return 0

        elif self.m_sysinfo_engine_type == "tvm":
            self.gen_tvm_code()
            self.make_deployment_yaml()  
            self.m_last_run_state = 0
            os.system("chmod -Rf 777 %s" % self.m_current_code_folder)
            return 0
        return 0

    ####################################################################
    def gen_pytorch_code(self):
        """
        convert pytorch template code

        Args: None
        Returns: int
            0 : success
            -1 : error
        """
        # self.m_sysinfo_libs = [python==3.8, torch>=1.1.0]
        #        self.m_sysinfo_apt = ['vim']
        #        self.m_sysinfo_papi = ['torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'albumentations']
        # yaml용 entry point, work dir, network, k8s 관련 항목 채우기
        # 가속기 고려 코드 생성
        # annotation 화일 복사하기
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)

        if self.m_deploy_type == 'cloud' or self.m_deploy_type == 'pc_server':
            self.m_sysinfo_libs = ['python==3.8', 'torch>=1.1.0']
            self.m_sysinfo_apt = ['vim']
            self.m_sysinfo_papi = ['flask', 'werkzeug', 'imageio', 'torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'pyyaml', 'json', 'albumentations', 'pathlib']
            self.m_deploy_entrypoint = ['output.py']
            # web폴더 복사후 .db화일 삭제 
            os.system("cp -r ./db/web/* %s" % self.m_current_code_folder) 
            os.system("rm %s/*.db" % self.m_current_code_folder) 
            # pt 화일 복사 done  
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            os.system("cp  %s  %s" % (pt_path, self.m_current_code_folder))
            # index.db 가속기 고려 코드 생성 후 index.py로 복사
            str = ''
            str += "def_port_num = %d\n" %  self.m_deploy_network_serviceport 
            # index.py 오픈 후 str 복사
            try: 
                outpath = "%s/%s" % (self.m_current_code_folder, "index.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("index.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/web/index.db", 'r')
            except IOError as err:
                logging.debug("web/index.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # pytorch-yolov9.db 가속기 고려 코드 생성 후 output.py로 복사
            str = ''
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            str += "def_pt_file = '%s'\n" % self.m_nninfo_weight_pt_file  
            if self.m_sysinfo_acc_type == "cuda":
                str += "def_dev = 'cuda:0'\n"
            elif self.m_sysinfo_acc_type == "opencl":
                str += "def_dev = 'opencl'\n"
            else:
                str += "def_dev = 'cpu'\n"
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/web/pytorch-yolov9.db", 'r')
            except IOError as err:
                logging.debug("web/pytorch-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file,  self.m_current_code_folder)) 
            # make requirement file in  code_folder 
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
        elif self.m_deploy_type == 'k8s':
            self.m_sysinfo_libs = ['python==3.8', 'torch>=1.1.0']
            self.m_sysinfo_apt = ['vim']
            self.m_sysinfo_papi = ['flask', 'werkzeug', 'imageio', 'torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'pyyaml', 'json', 'albumentations', 'pathlib']
            self.m_deploy_entrypoint = ['output.py']
            os.system("mkdir %s/fileset" % self.m_current_code_folder) 
            os.system("mkdir %s/fileset/yolov7" % self.m_current_code_folder) 
            k8s_path = "%s/fileset/yolov7" % self.m_current_code_folder
            os.system("cp -r ./db/web/* %s" % k8s_path) 
            # pt 화일 복사 done  
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            os.system("cp  %s  %s" % (pt_path, k8s_path))
            # index.db 가속기 고려 코드 생성 후 index.py로 복사
            str = ''
            # defalt k8s app service port 8902
            str += "def_port_num = %d\n" %  self.m_deploy_network_serviceport 
            # index.py 오픈 후 str 복사
            try: 
                outpath = "%s/%s" % (k8s_path, "index.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("index.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/web/index.db", 'r')
            except IOError as err:
                logging.debug("web/index.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # pytorch-yolov9.db 가속기 고려 코드 생성 후 output.py로 복사
            str = ''
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            str += "def_pt_file = '%s'\n" % self.m_nninfo_weight_pt_file  
            if self.m_sysinfo_acc_type == "cuda":
                str += "def_dev = 'cuda:0'\n"
            elif self.m_sysinfo_acc_type == "opencl":
                str += "def_dev = 'opencl'\n"
            else:
                str += "def_dev = 'cpu'\n"
            try:
                outpath = "%s/%s" % (k8s_path, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/web/pytorch-yolov9.db", 'r')
            except IOError as err:
                logging.debug("web/pytorch-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, k8s_path)) 
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            os.system("cp  %s  %s/fileset" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            # copy requirement file to code_folder just for testing
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
            # .db 지우기
            os.system("rm %s/*.db" % k8s_path) 
            # requirements.txt fileset/yolov7에 복사하기
            os.system("cp %s %s" % (outpath, k8s_path)) 
        else: # ondevice no need web 
            self.m_sysinfo_libs = ['python==3.8', 'torch>=1.1.0']
            self.m_sysinfo_apt = []
            self.m_sysinfo_papi = ['torch', 'torchvision', 'numpy', 'pathlib', 'opencv-python']
            self.m_deploy_entrypoint = ['output.py']
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            os.system("cp  %s  %s" % (pt_path, self.m_current_code_folder))
            str = ''
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            str += "def_pt_file= '%s'\n" % self.m_nninfo_weight_pt_file  
            if self.m_sysinfo_acc_type == "cuda":
                str += "def_dev = 'cuda:0'\n"
            elif self.m_sysinfo_acc_type == "opencl":
                str += "def_dev = 'opencl'\n"
            else:
                str += "def_dev = 'cpu'\n"
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/pytorch-yolov9.db", 'r')
            except IOError as err:
                logging.debug("pytorch-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file,  self.m_current_code_folder)) 
            # copy requirement file to code_folder just for testing
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
        return 0

    ####################################################################
    def gen_acl_code(self):
        """
        Generate template code for ARM ACL

        Args: None
        Returns: None
        """

        self.m_converted_file = "./db/yolo_v3_tiny_darknet_fp32.tflite"
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        if os.path.isfile(self.m_converted_file):
            shutil.copy(self.m_converted_file, self.m_current_code_folder)
        # after converting and copying, remove temporary converted file 

        try:
            f = open(self.get_code_filepath(self.m_deploy_python_file), 'w')
        except IOError as err:
            logging.debug("Python File Write Error")
            return -1

        # yolov3.head
        try:
            f1 = open("./db/yolov3.head", 'r')
        except IOError as err:
            logging.debug("yolov3 head open error")
            return -1
        for line1 in f1:
            f.write(line1)
        f1.close()

        # variable setting
        f.write('\ndef_model_file_path = "yolo_v3_tiny_darknet_fp32.tflite"\n')
        f.write('def_model_name = "yolo_v3_tiny"\n')
        f.write("def_preferred_backends = ['GpuAcc', 'CpuAcc', 'CpuRef']\n")
        # f.write("def_preferred_backends = ['CpuAcc', 'CpuRef']\n")
        f.write('def_profiling_enabled = "true"\n\n')
        # def_labels
        tmp_str = "def_labels = ["
        for cnt in range(0, self.m_nninfo_number_of_labels):
            if (cnt % 5) == 0:
                tmp_str = "%s%s%s%s" % (tmp_str, def_newline, def_4blank, def_4blank)
            if cnt == (self.m_nninfo_number_of_labels - 1):
                tmp_str = "%s%s%s%s" % (tmp_str, "'", self.m_nninfo_labelmap_info[cnt], "'")
            else:
                tmp_str = "%s%s%s%s%s" % (tmp_str, "'", self.m_nninfo_labelmap_info[cnt],
                                          "'", ', ')
        tmp_str = "%s%s" % (tmp_str, " ]\n\n")
        f.write(tmp_str)

        # yolov3.body
        try:
            f2 = open("./db/yolov3.body", 'r')
        except IOError as err:
            logging.debug("yolov3 body open error")
            return -1
        for line2 in f2:
            f.write(line2)
        f2.close()

        f.close()

        # manual copy
        if os.path.isfile(def_n2_manual):
            shutil.copy(def_n2_manual, self.m_current_code_folder)

        return


    def gen_trt_code(self):
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)

        if self.m_deploy_type == 'cloud' or self.m_deploy_type == 'pc_server':
            self.m_sysinfo_libs = ['python==3.8']
            self.m_sysinfo_apt = ['vim', 'tensorrt']
            self.m_sysinfo_papi = ['flask', 'werkzeug', 'imageio', 'torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'pyyaml', 'json', 'albumentations']
            self.m_deploy_entrypoint = ['output.py']
            # web폴더 복사후 .db화일 삭제 
            os.system("cp -r ./db/trtweb/* %s" % self.m_current_code_folder) 
            os.system("rm %s/*.db" % self.m_current_code_folder) 
            # onnx 화일 복사   
            onnx_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_onnx_file)
            os.system("cp  %s  %s" % (onnx_path, self.m_current_code_folder))
            # index.db 가속기 고려 코드 생성 후 index.py로 복사
            str = ''
            str += "def_port_num = %d\n" %  self.m_deploy_network_serviceport 
            # index.py 오픈 후 str 복사
            try: 
                outpath = "%s/%s" % (self.m_current_code_folder, "index.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("index.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/trtweb/index.db", 'r')
            except IOError as err:
                logging.debug("trtweb/index.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # tensorrt-yolov9.db 가속기 고려 코드 생성 후 output.py로 복사
            str = ''
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            a_file = def_trt_calib_cache.split("/")
            str += "def_calib_cache = '%s'\n" % a_file[-1] 
            str += "def_trt_engine = 'v9-16.trt'\n" 
            str += "def_trt_precision = 'fp16'\n" 
            str += "def_trt_max_detection = 100\n" 
            str += "def_onnx_model = '%s'\n" %  self.m_nninfo_weight_onnx_file
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/trtweb/tensorrt-yolov9.db", 'r')
            except IOError as err:
                logging.debug("web/tensorrt-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            #copy util file
            shutil.copy(def_trt_myutil_file_name, self.m_current_code_folder)
            #copy calib file
            shutil.copy(def_trt_calib_cache, self.m_current_code_folder)
            # copy requirement.txt
            # copy requirement file to code_folder just for testing
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
        elif self.m_deploy_type == 'k8s':
            os.system("mkdir %s/fileset" % self.m_current_code_folder) 
            os.system("mkdir %s/fileset/yolov7" % self.m_current_code_folder) 
            k8s_path = "%s/fileset/yolov7" % self.m_current_code_folder
            os.system("cp -r ./db/trtweb/* %s" % k8s_path) 
            os.system("rm %s/*.db" % k8s_path) 
            # onnx 화일 복사   
            onnx_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_onnx_file)
            os.system("cp  %s  %s" % (onnx_path, k8s_path))
            self.m_sysinfo_libs = ['python==3.8']
            self.m_sysinfo_apt = ['vim', 'tensorrt']
            self.m_sysinfo_papi = ['flask', 'werkzeug', 'imageio', 'torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'pyyaml', 'json', 'albumentations']
            self.m_deploy_entrypoint = ['output.py']
            # index.db 가속기 고려 코드 생성 후 index.py로 복사
            str = ''
            str += "def_port_num = %d\n" %  self.m_deploy_network_serviceport 
            # index.py 오픈 후 str 복사
            try: 
                outpath = "%s/%s" % (k8s_path, "index.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("index.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/trtweb/index.db", 'r')
            except IOError as err:
                logging.debug("trtweb/index.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            # tensorrt-yolov9.db 가속기 고려 코드 생성 후 output.py로 복사
            str = ''
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            a_file = def_trt_calib_cache.split("/")
            str += "def_calib_cache = '%s'\n" % a_file[-1] 
            str += "def_trt_engine = 'v9-16.trt'\n" 
            str += "def_trt_precision = 'fp16'\n" 
            str += "def_trt_max_detection = 100\n" 
            str += "def_onnx_model = '%s'\n" %  self.m_nninfo_weight_onnx_file
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            try:
                outpath = "%s/%s" % (k8s_path, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/trtweb/tensorrt-yolov9.db", 'r')
            except IOError as err:
                logging.debug("trtweb/tensorrt-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()

            #copy util file
            shutil.copy(def_trt_myutil_file_name, k8s_path)
            #copy calib file
            shutil.copy(def_trt_calib_cache, k8s_path)
            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, k8s_path)) 
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            os.system("cp  %s  %s/fileset" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            # copy requirement file to code_folder just for testing
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
            # .db 지우기
            os.system("rm %s/*.db" % k8s_path) 
            # requirements.txt fileset/yolov7에 복사하기
            os.system("cp %s %s" % (outpath, k8s_path)) 
        else: # ondevice no need web 
            self.m_sysinfo_libs = ['python==3.8']
            self.m_sysinfo_apt = ['vim', 'tensorrt']
            self.m_sysinfo_papi = ['flask', 'werkzeug', 'imageio', 'torch', 'torchvision', 'opencv-python', 'pandas', 'numpy', 'python-math', 'pyyaml', 'json', 'albumentations']
            self.m_deploy_entrypoint = ['output.py']
            # onnx 화일 복사   
            onnx_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_onnx_file)
            os.system("cp  %s  %s" % (onnx_path, self.m_current_code_folder))

            # tensorrt-yolov9.db 가속기 고려 코드 생성 후 output.py로 복사
            str = ''
            print(self.m_sysinfo_input_method)
            if type(self.m_sysinfo_input_method) != int:
                str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
            else:
                str += "def_input_location = %d\n" % self.m_sysinfo_input_method
            if type(self.m_sysinfo_output_method) != int:
                str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
            else:
                str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 
            str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
            str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
            a_file = def_trt_calib_cache.split("/")
            str += "def_calib_cache = '%s'\n" % a_file[-1] 
            str += "def_trt_engine = 'v9-16.trt'\n" 
            str += "def_trt_precision = 'fp16'\n" 
            str += "def_trt_max_detection = 100\n" 
            str += "def_onnx_model = '%s'\n" %  self.m_nninfo_weight_onnx_file
            str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, "output.py")
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("output.py file open error")
                self.m_last_run_state = 0
                return -1
            outf.write(str)
            try:
                inf = open("./db/tensorrt-yolov9.db", 'r')
            except IOError as err:
                logging.debug("tensorrt-yolov9.db open error")
                self.m_last_run_state = 0
                return -1
            for line1 in inf:
                outf.write(line1)
            inf.close()
            outf.close()
            # annotation 화일 복사 하기 done
            os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
            # copy requirement file to code_folder just for testing
            try:
                outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
                outf = open(outpath, "w")
            except IOError as err:
                logging.debug("requirements.txt file open error")
                self.m_last_run_state = 0
                return -1
            for item in self.m_sysinfo_papi:
                outf.write(item)
                outf.write('\n')
            outf.close()
            #copy util file
            shutil.copy(def_trt_myutil_file_name, self.m_current_code_folder)
            #copy calib file
            shutil.copy(def_trt_calib_cache, self.m_current_code_folder)
        return


    def gen_tvm_code(self):
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        onnx_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_onnx_file)
        onnx_model = onnx.load(onnx_path)
        input_name = "images"
        shape_dict = {input_name: [1, 3, def_TVM_width, def_TVM_height]}
        mod, params = tvm.relay.frontend.from_onnx(onnx_model, shape_dict)
        with open(def_TVM_mod, "w") as fo:
            fo.write(tvm.ir.save_json(mod))
        with open(def_TVM_param, "wb") as fo:
            fo.write(tvm.runtime.save_param_dict(params))
        self.m_nninfo_weight_onnx_file = [def_TVM_mod, def_TVM_param]
        os.system("cp %s  %s" % (def_TVM_mod, self.m_current_code_folder))
        os.system("cp %s  %s" % (def_TVM_param, self.m_current_code_folder))

        self.m_sysinfo_libs = ['python==3.8']
        self.m_sysinfo_apt = []
        self.m_sysinfo_papi = []
        self.m_deploy_entrypoint = ['output.py']
        str = ''
        str += "def_mod_path = 'yolov9-tvm.model'\n" 
        str += "def_param_path = 'yolov9-tvm.param'\n" 
        str += "def_label_yaml = '%s'\n" % self.m_nninfo_annotation_file 
        str += "def_conf_thres = %f\n" % self.m_nninfo_postproc_conf_thres
        str += "def_iou_thres = %f\n" % self.m_nninfo_postproc_iou_thres
        str += "def_dev_type = 'llvm'\n" 
        str += "def_data_type = 'float32'\n" 
        str += "def_width = 640\n" 
        str += "def_height = 640\n" 
        if type(self.m_sysinfo_input_method) != int:
            str += "def_input_location = '%s'\n" % self.m_sysinfo_input_method
        else:
            str += "def_input_location = %d\n" % self.m_sysinfo_input_method
        if type(self.m_sysinfo_output_method) != int:
            str += "def_output_location = '%s'\n"  % self.m_sysinfo_output_method 
        else:
            str += "def_output_location = %d\n"  % self.m_sysinfo_output_method 

        try:
            outpath = "%s/%s" % (self.m_current_code_folder, "output.py")
            outf = open(outpath, "w")
        except IOError as err:
            logging.debug("output.py file open error")
            self.m_last_run_state = 0
            return -1
        outf.write(str)
        try:
            inf = open("./db/tvm-yolov9.db", 'r')
        except IOError as err:
            logging.debug("tvm-yolov9.db open error")
            self.m_last_run_state = 0
            return -1
        for line1 in inf:
            outf.write(line1)
        inf.close()
        outf.close()

        # annotation 화일 복사 하기 done
        os.system("cp  %s  %s" % (self.m_nninfo_annotation_file, self.m_current_code_folder)) 
        # copy requirement file to code_folder just for testing
        try:
            outpath = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
            outf = open(outpath, "w")
        except IOError as err:
            logging.debug("requirements.txt file open error")
            self.m_last_run_state = 0
            return -1
        for item in self.m_sysinfo_papi:
            outf.write(item)
            outf.write('\n')
        outf.close()
        #copy util file
        shutil.copy(def_TVM_myutil_file_name, self.m_current_code_folder)
        return

    ####################################################################
    def clear(self):
        """
        Remove all generated files

        Args: None
        Returns: None
        """
        self.m_last_run_state = 0

        # check if run command was executed or not
        if self.m_current_userid == "":
            return

        if os.path.exists(self.m_current_code_folder):
            for file in os.scandir(self.m_current_code_folder):
                if os.path.isfile(file.path):
                    os.remove(file.path)
                else:
                    shutil.rmtree(file.path)

        if os.path.isfile(self.get_real_filepath(self.m_requirement_file)):
            os.remove(self.get_real_filepath(self.m_requirement_file))
        return

    ####################################################################
    def make_deployment_yaml(self):  # entry point , work dir
        if self.m_deploy_type == 'cloud':
            self.make_deploayment_cloud_yaml() 
        elif self.m_deploy_type == 'k8s':
            self.make_deploayment_k8s_yaml() 
        elif self.m_deploy_type == 'pc_web': 
            self.make_deploayment_web_yaml() 
        else:
            self.make_deploayment_ondevice_yaml() 
        return 0

    def make_deploayment_cloud_yaml(self): 
        #yaml
        t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
        t_com = {"custom_packages": t_pkg,
                 "libs": self.m_sysinfo_libs,
                 "engine": self.m_sysinfo_engine_type # added
                }
        t_build = {'architecture': self.m_sysinfo_cpu_type,
                   "accelerator": self.m_sysinfo_acc_type,
                   "os": self.m_sysinfo_os_type,
                   "image_uri": 'us-docker.pkg.dev/cloudrun/container/hello:latest',
                   "components": t_com }
        my_entry = ['run.sh', '-p', 'opt1', 'arg']
        t_deploy = {"type": 'docker',
                    "service_name": "hello",
                    # "workdir": "/workspace",
                    "entrypoint": self.m_deploy_entrypoint, 
                    "network": {"service_host_ip": self.m_deploy_network_hostip,
                                "service_host_port": self.m_deploy_network_hostport,
                                "service_container_port": self.m_deploy_network_serviceport}}
        t_total = {"build": t_build,
                "deploy": t_deploy}
        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_deployment_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        shutil.copy(r_file, self.m_current_file_path)
        return

    def make_deploayment_k8s_yaml(self): 
        t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
        t_com = {"custom_packages": t_pkg,
                 "libs": self.m_sysinfo_libs,
                 "engine": self.m_sysinfo_engine_type 
                }
        t_build = {'architecture': self.m_sysinfo_cpu_type,
                   "accelerator": self.m_sysinfo_acc_type,
                   "os": self.m_sysinfo_os_type,
                   "target_name": 'python:3.8',
                   "components": t_com
                   }
        t_deploy = {"type": 'docker',
                 "workdir": "/test/test",
                 'entrypoint': [ '/bin/bash', '-c'], 
                     "network": {
                     'service_host_ip': self.m_deploy_network_hostip,
                     "service_host_port": self.m_deploy_network_hostport,
                     "service_container_port": self.m_deploy_network_serviceport
                     },
                 "k8s": {
                     "nfs_ip": self.m_deploy_nfs_ip,
                     "nfs_path": self.m_deploy_nfs_path
                     }
             }
        a_file = self.m_nninfo_annotation_file.split("/")
        b_file = a_file[-1]
        t_opt = {"nn_file": 'output.py',
                 "weight_file": self.m_nninfo_weight_pt_file,
                 "annotation_file": b_file }
        t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}
        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_deployment_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        shutil.copy(r_file, self.m_current_file_path)
        shutil.copy(r_file, "%s/fileset" % self.m_current_code_folder)
        shutil.copy(r_file, "%s/fileset/yolov7" % self.m_current_code_folder)
        return

    def make_deploayment_web_yaml(self): 
        t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
        if self.m_sysinfo_engine_type == 'tensorrt':
            w_filw = self.m_nninfo_weight_onnx_file
            t_com = {"engine": 'tensorrt', "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        elif self.m_sysinfo_engine_type == 'tvm':
            w_filw = [def_TVM_lib_path, def_TVM_code_path]
            t_com = {"engine": 'tvm', "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        else:  # .pt file
            w_filw = self.m_nninfo_weight_pt_file
            t_com = {"engine": "pytorch", "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}

        t_build = {'architecture': self.m_sysinfo_cpu_type,
                   "accelerator": self.m_sysinfo_acc_type,
                   "os": self.m_sysinfo_os_type, "components": t_com}
        t_deploy = {"type": self.m_deploy_type, "work_dir": self.m_deploy_work_dir,
                    "entrypoint": self.m_deploy_python_file}
        a_file = self.m_nninfo_annotation_file.split("/")
        b_file = a_file[-1]
        t_opt = {"nn_file": self.m_deploy_python_file,
                 "weight_file": w_filw,
                 "annotation_file": b_file}
        t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}

        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_deployment_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        shutil.copy(r_file, self.m_current_file_path)
        return

    def make_deploayment_ondevice_yaml(self): 
        #yaml
        t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
        if self.m_sysinfo_engine_type == 'acl':
            w_filw = "yolo_v3_tiny_darknet_fp32.tflite"
            t_com = {"engine": "acl", "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        elif self.m_sysinfo_engine_type == 'tensorrt':
            w_filw = self.m_nninfo_weight_onnx_file
            t_com = {"engine": 'tensorrt', "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        elif self.m_sysinfo_engine_type == 'tvm':
            w_filw = [def_TVM_mod, def_TVM_param]
            t_com = {"engine": 'tvm', "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        else:  # .pt file
            w_filw = self.m_nninfo_weight_pt_file
            t_com = {"engine": "pytorch", "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}

        t_build = {'architecture': self.m_sysinfo_cpu_type,
                   "accelerator": self.m_sysinfo_acc_type,
                   "os": self.m_sysinfo_os_type, "components": t_com}
        t_deploy = {"type": self.m_deploy_type, "work_dir": self.m_deploy_work_dir,
                    "entrypoint": self.m_deploy_python_file}
        a_file = self.m_nninfo_annotation_file.split("/")
        b_file = a_file[-1]
        t_opt = {"nn_file": self.m_deploy_python_file,
                 "weight_file": w_filw,
                 "annotation_file": b_file }
        t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}

        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_deployment_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        shutil.copy(r_file, self.m_current_file_path)
        return

        
    ####################################################################
    def response(self):
        """
        Send Success Message to Project Manager 

        Args: None
        Returns: None 
        """
        try:
            host = socket.gethostbyname('projectmanager')
        except socket.gaierror:
            host = ''
        if host == '':
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                host = s.getsockname()[0]
            except socket.error as err:
                logging.debug("err")
                print("err")
        prj_url = "%s%s%s" % ('http://', host, ':8085/status_report')

        # add result code
        prj_url = "%s?container_id=code_gen&user_id=%s&project_id=%s" % (prj_url, self.m_current_userid, self.m_current_projectid)
        if self.m_last_run_state == 0:  # success
            prj_url = "%s&status=success" % prj_url 
        else:
            prj_url = "%s&status=failed" % prj_url 

        headers = {
            'Host': '0.0.0.0:8085',
            'Origin': 'http://0.0.0.0:8888',
            'Accept': "application/json, text/plain",
            'Access-Control_Allow_Origin': '*',
            'Access-Control-Allow-Credentials': "true",
            'vary': 'origin',
            'referrer-policy': 'same-origin'
            # 'Content-type': 'applocation/json'
            }
        try:
            ret = requests.get(url=prj_url, headers=headers)
            # ret = requests.get(url=prj_url, headers=headers, params=prj_data)
        except requests.exceptions.HTTPError as err:
            logging.debug("Http Error:")
            print("Http Error:")
        except requests.exceptions.ConnectionError as err:
            logging.debug("Error Connecting:")
            print("Error Connecting:")
        except requests.exceptions.Timeout as err:
            logging.debug("Timeout Error:")
            print("Timeout Error:")
        except requests.exceptions.RequestException as err:
            logging.debug("OOps: Something Else")
            print("OOps: Something Else")
        logging.debug(prj_url)
        logging.debug("response for report")
        return


####################################################################
####################################################################
class MyHandler(SimpleHTTPRequestHandler):
    """Webserver Definition """
    
    m_flag = 1
    m_stop = 0
    m_obj = 0
    # allowed_list = ('0,0,0,0', '127.0.0.1')

    @staticmethod
    def set_obj(cobj):
        MyHandler.m_obj = cobj
        return

    def send_cors_headers(self):
        """
        send header for CORS

        Args: None
        Returns: None
        """
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header("Access-Control-Allow-Methods", "GET, OPTION")
        self.send_header("Access-Control-Allow-Credentials", "true")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Origin, Accept, token")
        self.send_header("vary", "origin")

    def do_GET(self):
        """
        process HTTP GET command

        Args: None
        Returns: None
        """
        # (c_host, c_port) = self.client_address
        # if c_host not in self.allowed_list:
        #     self.send_response(401, 'request not allowed')
        #     return

        logging.debug("code_gen: GET method called !!!")
        if self.path[1] == '?':
            t_path = "%s%s" % ('/', self.path[2:])
        else:
            t_path = self.path
        pathlist = t_path.split('/')[1].split('?')
        cnt = len(pathlist)
        if cnt < 2:
            cmd = "unknown"
        else:
            ctmp = pathlist[1].split('&')
            mycnt = len(ctmp)
            if mycnt == 0:
                cmd = pathlist[0]
            elif mycnt == 1:
                cmd = pathlist[0]
                userid = ctmp[0].split('user_id')[1].split('=')[1]
                if userid == '""':
                    self.m_obj.set_folder("", "")
                else:
                    self.m_obj.set_folder(userid, "")
            else:  # mycnt == 2:
                cmd = pathlist[0]
                userid = ctmp[0].split('user_id')[1].split('=')[1]
                prjid = ctmp[1].split('project_id')[1].split('=')[1]
                if userid == '""' or userid == '%22%22':
                    userid = ""
                if prjid == '""' or prjid == '%22%22':
                    prjid = ""
                self.m_obj.set_folder(userid, prjid)
        logging.debug("code_gen: cmd = %s" %  cmd)

        if cmd == "start":
            buf = '"started"'
            self.send_response(200, 'OK')
            self.send_cors_headers()
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
            logging.debug("code_gen: send_ack")

            if self.m_flag == 1:
                # self.m_obj.run()
                with self.m_obj.lock:
                    self.m_obj.the_cnt = self.m_obj.the_cnt + 1
                self.m_obj.ev.set()
                logging.debug("event Set")
            # send notice to project manager
            # self.m_obj.response()
            # print("code_gen: send_status_report to manager")
        elif cmd == 'stop':
            buf = '"finished"'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
            self.m_obj.clear()
            self.m_stop = 1
        elif cmd == "clear":
            self.m_obj.clear()
            buf = '"OK"'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
        elif cmd == "pause":
            self.m_flag = 0
            buf = '"OK"'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
        elif cmd == '"resume"':
            self.m_flag = 1
            buf = "OK"
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
        elif cmd == 'status_request':
            logging.debug("status_request called")
            buf = '"failed"'
            if self.m_obj.m_current_userid == "":
                buf = '"ready"'
            else:
                if self.m_obj.m_atwork == 1:
                    buf = '"running"'
                else:
                    if self.m_flag == 0:
                        buf = '"stopped"'
                    else:
                        buf = '"completed"'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())
            logging.debug("status_request response = %s" % buf)
        else:
            buf = '""'
            self.send_response(200, 'ok')
            self.send_cors_headers()
            self.send_header('Content-Type', 'text/plain')
            self.send_header("Content-Length", "%d" % len(buf))
            self.end_headers()
            self.wfile.write(buf.encode())

        if self.m_stop == 1:
            time.sleep(1)
            raise KeyboardInterrupt  # to finish web sever
        return
        
    def do_OPTIONS(self):
        """
        send header for CORS
        
        Args: None
        Returns: None
        """
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    '''
    def do_POST(self):
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        dataLength = int(self.headers["Content-Length"])
        data = self.rfile.read(dataLength)
        print(data)
        response = {"status": "OK"}
        self.send_dict_response(response)
        return
    '''


####################################################################
####################################################################
if __name__ == '__main__':
    # just for test only
    # print("DEBUG")
    # tmp = CodeGen()
    # tmp.set_folder('jammanbo', '1234')
    # tmp.run()
    # exit()

    m_obj = CodeGen()
    mythr = threading.Thread(target=m_obj.thread_for_run, daemon=True, name="MyThread")
    mythr.start()

    MyHandler.set_obj(m_obj)
    server = HTTPServer(('', def_codegen_port), MyHandler)
    logging.debug("Started WebServer on Port %d" % def_codegen_port)
    logging.debug("Press ^C to quit WebServer")

    try:
        server.serve_forever()
    except KeyboardInterrupt as e:
        time.sleep(1)
        server.socket.close()
        logging.debug("wait for thread done")
        m_obj.wait_for_done()


#스트링으로 함수 호출하기 #1
#file name이 user.py class가 User라 가정
'''
user.py의 내용
class User():
    name = 'abc'
    def doSomething():
        print(name)
from    user    import  User as user
doSomething = getattr(user, 'DoSomething')
doSemethins(user)

#스트링으로 함수 호출하기 #2
locals()[String 변수]()
locals()["myFunction"]()
globals()["myFunction"]()
'''
