"""
copyright notice
"""

"""
code_gen.py
This module generates template code for RKNN, PyTorch, and ArmNN based on AutoNN's output.
"""
# khlee To Do List
'''
web app template gen.
nms code gen.
deployment.yaml location change : under nn_model
deployment.yaml user editing -> user can be add papy app libs 
tflite code gen.
remove rknn
input source mp4 + jpg in same folder
'''

import os
import socket
import shutil
import zipfile
import numpy as np
import onnx
# for test import tvm
# for test import tvm.relay as relay
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

# for rknn
#from rknn.api import RKNN
def_rknn_file = "yolov5.rknn"
def_rknn_template = "yolov5.template"

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
def_TVM_data_type = "float16"
def_TVM_lib_path = "mylib.so"
def_TVM_code_path = "mycode.bin"
def_TVM_myutil_file_name = "./db/myutil.py"


# defualt values
def_nninfo_file = "neural_net_info.yaml"
def_sysinfo_file = "project_info.yaml"
def_requirement_file = "deployment.yaml"

def_task_type = 'detection'  # classification
def_memory_size = 1
def_cpu_type = ""  # 'x86'  # arm
def_acc_type = ""  # 'cpu'  # cpu/cuda/opencl
def_os_type = ""  # ubuntu'  # linux/windows
def_engine_type = ""  # pytorch'  # acl/rknn/tvm/tensorrt
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
def_weight_onnx_file = "tango.onnx"  # ' input.onnx'
def_dataset_path = "/tango/datasets"  
def_annotation_file = "dataset.yaml"  # 'coco.dat'
def_ondevice_python_requirements = "requirements.txt"  

def_newline = '\n'
def_4blank = "    "

def_codegen_port = 8888

def_n2_manual = './db/odroid-n2-manual.txt'
def_m1_manual = './db/odroid-m1-manual.txt'
def_tensorrt_manual = './db/odroid-m1-manual.txt'
def_tvm_manual = './db/odroid-m1-manual.txt'

def_yolo_base_file_path = "."
def_yolo_requirements = "yoloe_requirements.txt"


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
    m_nninfo_preproc_norm = []
    m_nninfo_preproc_mean = []
    m_nninfo_output_format_allow_list = False
    m_nninfo_output_number = 0
    m_nninfo_output_size = []
    m_nninfo_output_pred_format = []
    m_nninfo_postproc_conf_thres = 0
    m_nninfo_postproc_iou_thres = 0
    m_nninfo_postproc_need_nms = False

    m_nninfo_input_tensor_shape = []
    m_nninfo_input_data_type = ""
    m_nninfo_yolo_base_file_path = def_yolo_base_file_path 
    m_nninfo_yolo_require_packages = []
    m_nninfo_user_libs = []
    m_py = ""
    m_pt_model = ""

    m_converted_file = def_rknn_file
    m_deploy_python_file = def_deploy_python_file
    m_requirement_file = def_requirement_file

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
                    # khlee 
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
    def add_user_libs(self, libs):
        """
        add lib that user added for the code that user edited
        Args:
            libs : list of libs 
        Returns: int
            0 : success
            -1 : file error
        """
        if isinstance(libs, list):
            for item in libs:
                if not libs in self.m_nninfo_user_libs:
                    self.m_nninfo_user_libs.append(libs)
        else:
            if not libs in self.m_nninfo_user_libs:
                self.m_nninfo_user_libs.append(libs)
                
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
        print("aaa")
        print(pid)
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
                self.m_sysinfo_input_method = value  # url, path, camera ID 
            elif key == 'output_method':
                # 0:graphic, 1:text, path, url
                self.m_sysinfo_output_method = value  
            elif key == 'user_editing':
                self.m_sysinfo_user_editing = value  # yes/no
            elif key == 'confidence_thresh':
                self.m_sysinfo_confidence_thresh = value  
            elif key == 'iou_thresh':
                self.m_sysinfo_iou_thresh = value  
            elif key == 'dataset':
                myset = "%s/%s.yaml" % (value, value)
                self.m_nninfo_annotation_file = myset
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
            logging.debug("Sysinfo file open error!!")
            return -1
        m_nninfo = yaml.load(f, Loader=yaml.FullLoader)

        # parse nninfo
        for key, value in sorted(m_nninfo.items()):
            if key == 'base_dir_autonn':
                self.m_nninfo_yolo_base_file_path = value
                # add python code path
                my_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_yolo_base_file_path)
                if not my_path in sys.path:
                    sys.path.append(my_path)
                    sys.path.append(self.m_current_file_path)
            if key == 'class_file':
                # subval = value.get('os')
                self.m_nninfo_class_file = value
                if len(value) == 1:
                    self.m_nninfo_model_definition_file = value
                else:
                    self.m_nninfo_model_definition_file = value[0]
            elif key == 'class_name':
                self.m_nninfo_class_name = value
            elif key == 'weight_file':
                if isinstance(value, list):
                    cnt = len(value)
                    for i in range(cnt):
                        tmp_str = value[i]
                        # .pt
                        val = ".pt" in tmp_str
                        if val:
                            self.m_nninfo_weight_pt_file = tmp_str
                        else:
                            self.m_nninfo_weight_onnx_file = tmp_str  # .onnx
                else:
                    self.m_nninfo_weight_pt_file = value
                    # self.m_nninfo_weight_onnx_file = value
            elif key == 'label_info_file':
                self.m_nninfo_annotation_file = value
                # parsing  labelmap
                try:
                    annotation_file = "%s/%s" % (def_dataset_path, self.m_nninfo_annotation_file)
                    f1 = open(annotation_file, encoding='UTF-8')
                except IOError as err:
                    logging.debug("LabelMap file open error!!")
                    return -1
                labelinfo = yaml.load(f1, Loader=yaml.FullLoader)
                for key1, value1 in sorted(labelinfo.items()):
                    if key1 == 'names':
                        self.m_nninfo_labelmap_info = value1
                    elif key1 == 'nc':
                        self.m_nninfo_number_of_labels = int(value1)

            elif key == 'input_tensor_shape':  # [1,2,3,4] -> integer conversion needed
                if isinstance(value, list):
                    cnt = len(value)
                    tmp1 = list(range(0, cnt))
                    for i in range(cnt):
                        tmp1[i] = int(value[i])
                    self.m_nninfo_input_tensor_shape = tmp1
            elif key == 'input_data_type':  # fp32
                self.m_nninfo_input_data_type = value
            elif key == 'anchors':
                t_cnt = len(value)
                cord = 0
                mask = []
                my_anchor = []
                for element in range(0, t_cnt):
                    a1 = []
                    ttmp = value[element]
                    tmp_len = len(ttmp)
                    ele_num = int(tmp_len / 2)
                    m1 = []
                    ttt = 0
                    for n2 in range(0, ele_num):
                        m1.append(cord)
                        cord = cord + 1
                        a1.append([ttmp[ttt], ttmp[ttt + 1]])
                        ttt = ttt + 2
                    mask.append(m1)
                    my_anchor.extend(a1)
                self.m_nninfo_anchors = my_anchor
                self.m_nninfo_mask = mask
            elif key == 'vision_lib':  # cv2
                self.m_sysinfo_vision_lib = value
            elif key == 'norm':
                self.m_nninfo_preproc_norm = value
            elif key == 'mean':
                self.m_nninfo_preproc_mean = value
            elif key == 'output_format_allow_list':
                self.m_nninfo_output_format_allow_list = value
            elif key == 'output_number':
                self.m_nninfo_output_number = int(value)
            elif key == 'output_size':
                self.m_nninfo_output_size = value
            elif key == 'output_pred_format':
                self.m_nninfo_output_pred_format = value
            elif key == 'conf_thres':
                self.m_nninfo_postproc_conf_thres = float(value)
            elif key == 'iou_thres':
                self.m_nninfo_postproc_iou_thres = float(value)
            elif key == 'need_nms':
                self.m_nninfo_postproc_need_nms = value
            self.m_nninfo_yolo_require_packages = []
            try:
                f = open(self.get_real_filepath(def_yolo_requirements)) 
                while True:
                    line = f.readline()
                    if not line:
                        break
                    self.m_nninfo_yolo_require_packages.append(line.replace("\n", ""))
                f.close()
            except IOError as err:
                print("yolo requirement file not found")
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
        self.parse_nninfo_file()
        self.parse_sysinfo_file()

        # if there are files in the target folder, remove them
        self.clear()

        if self.m_sysinfo_task_type == "classification":
            if self.m_sysinfo_engine_type == 'pytorch':
                self.m_sysinfo_libs = []
                self.m_sysinfo_apt = ['vim', 'python3.9']
                self.m_sysinfo_papi = ['torch', 'torchvision', 'pandas', 'numpy', 'albumentations']
                self.m_deploy_entrypoint = self.m_deploy_python_file
                # code gen for classificaaion
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
                f.write("DEF_ACC = %s\n" % "cpu") # only for testing self.m_sysinfo_acc_type) 
                f.write("DEF_PT_FILE = %s\n\n\n" % self.m_nninfo_weight_pt_file)
                try:
                    f1 = open("./db/resnet152.db", 'r')
                except IOError as err:
                    logging.debug("resnet162 db open error")
                    return -1
                for line1 in f1:
                    f.write(line1)
                f1.close()
                f.close()

                if self.m_deploy_type == 'cloud':
                    self.make_requirements_file_for_docker()
                elif self.m_deploy_type == 'k8s':
                    self.make_requirements_file_for_docker()
                elif self.m_deploy_type == 'pc_server':
                    self.make_requirements_file_for_PCServer()
                elif self.m_deploy_type == 'pc_web':
                    self.make_requirements_file_for_PCServer()
                else:
                    self.make_requirements_file_for_others()
            elif self.m_sysinfo_engine_type == 'tensorrt':
                # add code for tensorrt
                if not os.path.exists(self.m_current_code_folder):
                    os.makedirs(self.m_current_code_folder)
                # read model file
                # convert the model file to onnx file
                # convert the onnx file to tensorrt file
                # copy tensorrt file to the nn_model folder
                # copy template code file to the nn_model folder

                if self.m_deploy_type == 'cloud':
                    self.make_requirements_file_for_docker()
                elif self.m_deploy_type == 'k8s':
                    self.make_requirements_file_for_docker()
                elif self.m_deploy_type == 'pc_server':
                    self.make_requirements_file_for_PCServer()
                elif self.m_deploy_type == 'pc_web':
                    self.make_requirements_file_for_PCServer()
                else:
                    self.make_requirements_file_for_others()
            else:
                print("the inference engine is not support for classification")
            self.m_last_run_state = 0
            return

        # code gen
        # khlee rknn must be deleted !!!!!!
        if self.m_sysinfo_engine_type == 'rknn':   
            self.m_sysinfo_libs = ['libprotobuf.so.10', 'x86_64-linux-gnu']
            self.m_sysinfo_apt = ['python3==3.6', 'python3-dev', 'python3-pip',
                                  'libxslt1-dev', 'zlib1g-dev', 'libglib2.0',
                                  'libsm6', 'libgl1-mesa-glx', 'libprotobuf-dev', 'gcc']
            self.m_sysinfo_papi = [
                'rknn_toolkit2-1.2.0_f7bb160f-cp36-cp36m-linux_x86_64.whl',
                'numpy==1.16.6',
                'onnx==1.7.0',
                'onnxoptimizer==0.1.0',
                'onnxruntime==1.6.0',
                'tensorflow =1.14.0',
                'tensorboard==1.14.0',
                'protobuf==3.12.0',
                'torch==1.6.0',
                'torchvision==0.7.0',
                'psutil==5.6.2',
                'ruamel.yaml==0.15.81',
                'scipy=1.2.1',
                'tqdm==4.27.0',
                'requests==2.21.0',
                'opencv-python==4.4.0.46',
                'PuLP==2.4',
                'scikit_image==0.17.2',
                'flatbuffers==1.12'
            ]
            self.m_deploy_entrypoint = [self.m_deploy_python_file]
            self.gen_rknn_python_code()
            self.make_requirements_file_for_others()

        # python
        elif self.m_sysinfo_engine_type == 'pytorch':
            self.m_sysinfo_libs = []
            self.m_sysinfo_apt = ['vim', 'python3.9']
            self.m_sysinfo_papi = []
            self.m_deploy_entrypoint = self.m_deploy_python_file
            self.gen_python_code()
            if self.m_deploy_type == 'cloud':
                # kkkhhhllleeeeeeeee khlee
                self.m_nninfo_weight_pt_file = 'yolov3.pt',
                self.make_requirements_file_for_docker()
            elif self.m_deploy_type == 'k8s':
                # kkkhhhllleeeeeeeee khlee
                self.m_nninfo_weight_pt_file = "yolov5s.pt"
                self.make_requirements_file_for_docker()
            elif self.m_deploy_type == 'pc_server':
                self.make_requirements_file_for_PCServer()
            elif self.m_deploy_type == 'pc_web':
                self.make_requirements_file_for_PCServer()
            else:
                self.make_requirements_file_for_others()

        # acl
        elif self.m_sysinfo_engine_type == 'acl':
            self.m_deploy_entrypoint = [self.m_deploy_python_file]
            self.m_sysinfo_libs = ['mali-fbdev']
            self.m_sysinfo_apt = ['clinfo', 'ocl-icd-libopnecl1',
                                  'ocl-icd-opencl-dev', 'python3-opencv', 'python3-pip']
            self.m_sysinfo_papi = []
            self.gen_acl_code()
            self.make_requirements_file_for_others()
        elif self.m_sysinfo_engine_type == 'tflite':
            # pt 2 onnx
            if self.m_nninfo_yolo_base_file_path != ".":  
                if isinstance(self.m_nninfo_class_file, list):
                    m_filelist = self.m_nninfo_class_file
                else:
                    m_filelist = [self.m_nninfo_class_file]
                for mfile in m_filelist:
                    self.copy_subfolderfile(mfile, self.m_nninfo_yolo_base_file_path)

            # convert .py to onnx and copy
            if isinstance(self.m_nninfo_class_file, list):
                t_file = self.m_nninfo_class_file[0]
            else:
                t_file = self.m_nninfo_class_file
            t_file = "%s%s" % (t_file, "\n")
            tmp = t_file.split("/")
            cnt = len(tmp)
            t_file = tmp[0]
            for  i in range(cnt-1):
                t_file = "%s.%s" % (t_file, tmp[i+1])
            t_file = t_file.split(".py\n")[0]
            t_file = t_file.split("\n")[0]
            t_split = self.m_nninfo_class_name.split("(")
            t_class_name = t_split[0]
            tmp_param = t_split[1].split(")")[0]
            tmp_param = tmp_param.split("=")[1]

            t_list = t_file.split(".")
            t_list_len = len(t_list)
            t_fromlist = ""
            for i in range(t_list_len-1):
                t_fromlist = "%s%s" % (t_fromlist, t_list[i])
                if i < (t_list_len-2): 
                    t_fromlist = "%s%s" % (t_fromlist, ".")
            t_impmod = __import__(t_file, fromlist=[t_fromlist])
            t_cls = getattr(t_impmod, t_class_name)

            f_param = self.get_real_filepath(tmp_param[1:-1])
            pt_model = t_cls(cfg=f_param)
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            pt_model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')), strict=False)
            pt_model.eval()
            # convert and save onnx file 
            dummy = torch.randn(
                    self.m_nninfo_input_tensor_shape[0], 
                    self.m_nninfo_input_tensor_shape[1], 
                    self.m_nninfo_input_tensor_shape[2], 
                    self.m_nninfo_input_tensor_shape[3], 
                    requires_grad=True)
            t_folder = "%s/%s" % (self.m_current_code_folder, self.m_nninfo_weight_onnx_file)
            torch.onnx.export(pt_model, dummy,
                    t_folder,
                    opset_version = 11,
                    export_params=True, 
                    input_names=['input'],
                    output_names=['output'])
            print("tflite: pt -> onnx")
            # shutil.copy("./db/yolov7-tiny.onnx", t_folder)
            # onnx 2 tf
            os.system('onnx-tf convert -i %s -o  mymodel.pb' % t_folder)
            print("tflite: onnx -> tf")
            # tf to tflite
            converter = tf.lite.TFLiteConverter.from_saved_model('mymodel.pb')
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("tflite: tf -> tflite")
            tf_lite_model = converter.convert()
            open('mymodel.tflite', 'wb').write(tf_lite_model)
            # /bin/rm -r mymodel.pb
            if os.path.exists("mymodel.pb"):
                shutil.rmtree("mymodel.pb")
            # rm mymodel.onnx
            os.remove(t_folder)
            # cp mymodel.tflite
            shutil.copy("mymodel.tflite", "/app/tflite_yolov7_test/app/main/assets/yolovy_tiny_fp32_320.tflite")
            # rm mymodel.tflite
            os.remove("mymodel.tflite")
            # build apk
            print("tflite: butild apk")
            if os.path.exists("/bin/rm -rf /app/tflite_yolov7_test/app/.cxx"):
                shutil.rmtree("/bin/rm -rf /app/tflite_yolov7_test/app/.cxx")
            os.system("/app/tflite_yolov7_test/gradlew init < /app/enter.txt  && cd /app/tflite_yolov7_test && ./gradlew assembleDebug")
            # copy apk nn_model
            print("tflite: copy apk")
            shutil.copy("/app/tflite_yolov7_test/app/build/outputs/apk/debug/app-debug.apk", self.m_current_code_folder)
            # copy android project files nn_model
            print("tflite: copy android project")
            shutil.copytree("/app/tflite_yolov7_test", self.m_current_code_folder)
            # kkkhhhllleeeeeeeee

        elif self.m_sysinfo_engine_type == "tensorrt":
            self.m_deploy_entrypoint = [self.m_deploy_python_file]
            # move autonn's pytorch to target folder(like prefix folder)
            if self.m_nninfo_yolo_base_file_path != ".":  
                if isinstance(self.m_nninfo_class_file, list):
                    m_filelist = self.m_nninfo_class_file
                else:
                    m_filelist = [self.m_nninfo_class_file]
                for mfile in m_filelist:
                    self.copy_subfolderfile(mfile, self.m_nninfo_yolo_base_file_path)

            # convert .py to onnx and copy
            if isinstance(self.m_nninfo_class_file, list):
                t_file = self.m_nninfo_class_file[0]
            else:
                t_file = self.m_nninfo_class_file
            t_file = "%s%s" % (t_file, "\n")
            tmp = t_file.split("/")
            cnt = len(tmp)
            t_file = tmp[0]
            for  i in range(cnt-1):
                t_file = "%s.%s" % (t_file, tmp[i+1])
            t_file = t_file.split(".py\n")[0]
            t_file = t_file.split("\n")[0]
            t_split = self.m_nninfo_class_name.split("(")
            t_class_name = t_split[0]
            tmp_param = t_split[1].split(")")[0]
            tmp_param = tmp_param.split("=")[1]

            t_list = t_file.split(".")
            t_list_len = len(t_list)
            t_fromlist = ""
            for i in range(t_list_len-1):
                t_fromlist = "%s%s" % (t_fromlist, t_list[i])
                if i < (t_list_len-2): 
                    t_fromlist = "%s%s" % (t_fromlist, ".")
            t_impmod = __import__(t_file, fromlist=[t_fromlist])
            t_cls = getattr(t_impmod, t_class_name)

            f_param = self.get_real_filepath(tmp_param[1:-1])
            pt_model = t_cls(cfg=f_param)
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            pt_model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')), strict=False)
            pt_model.eval()
            # convert and save onnx file 
            dummy = torch.randn(
                    self.m_nninfo_input_tensor_shape[0], 
                    self.m_nninfo_input_tensor_shape[1], 
                    self.m_nninfo_input_tensor_shape[2], 
                    self.m_nninfo_input_tensor_shape[3], 
                    requires_grad=True)
            t_folder = "%s/%s" % (self.m_current_code_folder, self.m_nninfo_weight_onnx_file)
            torch.onnx.export(pt_model, dummy,
                    t_folder,
                    opset_version = 11,
                    export_params=True, 
                    input_names=['input'],
                    output_names=['output'])
            # khlee only for test copy yolov.onnx to tango.onnx
            # must be deleted 
            shutil.copy("./db/yolov7.onnx", t_folder)

            self.gen_tensorrt_code(self.m_nninfo_input_tensor_shape[1], 
                    self.m_nninfo_input_tensor_shape[2], 
                    self.m_nninfo_input_data_type) 
            self.make_requirements_file_for_others()
        elif self.m_sysinfo_engine_type == "tvm":
            self.m_deploy_entrypoint = [self.m_deploy_python_file]
            tvm_dev_type = def_TVM_dev_type    # 0 llvm ,1 cuda,  
            tvm_width = def_TVM_width  
            tvm_height = def_TVM_height  
            tvm_data_type = def_TVM_data_type 
            # move autonn's pytorch to target folder(like prefix folder)
            if self.m_nninfo_yolo_base_file_path != ".":  
                if isinstance(self.m_nninfo_class_file, list):
                    m_filelist = self.m_nninfo_class_file
                else:
                    m_filelist = [self.m_nninfo_class_file]
                for mfile in m_filelist:
                    self.copy_subfolderfile(mfile, self.m_nninfo_yolo_base_file_path)

            # convert .py to onnx and copy
            if isinstance(self.m_nninfo_class_file, list):
                t_file = self.m_nninfo_class_file[0]
            else:
                t_file = self.m_nninfo_class_file
            t_file = "%s%s" % (t_file, "\n")
            tmp = t_file.split("/")
            cnt = len(tmp)
            t_file = tmp[0]
            for  i in range(cnt-1):
                t_file = "%s.%s" % (t_file, tmp[i+1])
            t_file = t_file.split(".py\n")[0]
            t_file = t_file.split("\n")[0]
            t_split = self.m_nninfo_class_name.split("(")
            t_class_name = t_split[0]
            tmp_param = t_split[1].split(")")[0]
            tmp_param = tmp_param.split("=")[1]

            t_list = t_file.split(".")
            t_list_len = len(t_list)
            t_fromlist = ""
            for i in range(t_list_len-1):
                t_fromlist = "%s%s" % (t_fromlist, t_list[i])
                if i < (t_list_len-2): 
                    t_fromlist = "%s%s" % (t_fromlist, ".")
            t_impmod = __import__(t_file, fromlist=[t_fromlist])
            t_cls = getattr(t_impmod, t_class_name)

            f_param = self.get_real_filepath(tmp_param[1:-1])
            pt_model = t_cls(cfg=f_param)
            pt_path = "%s/%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            pt_model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')), strict=False)
            pt_model.eval()
            # convert and save onnx file 
            dummy = torch.randn(
                    self.m_nninfo_input_tensor_shape[0], 
                    self.m_nninfo_input_tensor_shape[1], 
                    self.m_nninfo_input_tensor_shape[2], 
                    self.m_nninfo_input_tensor_shape[3], 
                    requires_grad=True)
            torch.onnx.export(pt_model, dummy,
                    self.get_real_filepath(self.m_nninfo_weight_onnx_file),
                    opset_version = 11,
                    export_params=True, 
                    input_names=['input'],
                    output_names=['output'])
            # khlee only for test copy yolov.onnx to tango.onnx
            # must be deleted
            # shutil.copy("./db/yolov7.onnx", self.get_real_filepath(self.m_nninfo_weight_onnx_file))
            # shutil.copy("./db/yolov7tvm.onnx", self.get_real_filepath(self.m_nninfo_weight_onnx_file))


            onnx_model = onnx.load(self.get_real_filepath(self.m_nninfo_weight_onnx_file))
            input_name = onnx_model.graph.input[0].name
            tensor_type = onnx_model.graph.input[0].type.tensor_type
            tmp_list = []
            if (tensor_type.HasField("shape")):
                for d in tensor_type.shape.dim:
                    if (d.HasField("dim_value")):
                        tmp_list.append(d.dim_value)
                i_shape = tuple(tmp_list)
                logging.debug(i_shape)
                (x_, y_, self.width, self.height) = i_shape
            else:
                i_shape = (1, 1, 224, 224)

            dtype = 'object'
            # check elem_type value
            if (tensor_type.HasField("elem_type")):
                tp = tensor_type.elem_type
                if tp == 1: # FLOAT
                    dtype = 'float32'
                elif tp == 2: # UINIT8
                    dtype = 'uint8'
                elif tp == 3: # INT8
                    dtype = 'int8'
                elif tp == 4: # UINT16
                    dtype = 'uint16'
                elif tp == 5: # INT16
                    dtype = 'int16'
                elif tp == 6: # INT32
                    dtype = 'int32'
                elif tp == 7: # INT64
                    dtype = 'int64'
                elif tp == 8: # STRING
                    dtype = 'unicode'
                elif tp == 9: # BOOL
                    dtype = 'bool'
                elif tp == 10: # FLOAT16
                    dtype = 'float16'
                elif tp == 11: # DOUBLE
                    dtype = 'double'
                elif tp == 12: # UINT32
                    dtype = 'uint32'
                elif tp == 13: # UINT64
                    dtype = 'uint64'
                elif tp == 14: # COMPLEX64
                    dtype = 'complex64'
                elif tp == 15: # COMPLEX128
                    dtype = 'complex128'
                elif tp == 16: # BFLOAT16
                    dtype = 'bfloat16'
            logging.debug(dtype)
    
            if tvm_dev_type == 0:
                target = "llvm"
            elif tvm_dev_type == 1:
                target = "cuda"
            elif tvm_dev_type == 2:
                target = "llvm -mtriple=aarch64-linux-gnu"
            else:
                target = "opencl"
            logging.debug(target)
            shape_dict = {input_name: i_shape}
            logging.debug(shape_dict)
            mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
            executable = relay.vm.compile(mod, target=target, params=params)
            code, lib = executable.save()
            lib.export_library(def_TVM_lib_path)
            with open(def_TVM_code_path, "wb") as outf:
                outf.write(code)
            if os.path.isfile(def_TVM_lib_path):
                shutil.copy(def_TVM_lib_path, self.m_current_code_folder)
                # after converting and copying, remove temporary converted file 
                os.remove(def_TVM_lib_path)
            if os.path.isfile(def_TVM_code_path):
                shutil.copy(def_TVM_code_path, self.m_current_code_folder)
                # after converting and copying, remove temporary converted file 
                os.remove(def_TVM_code_path)
            # copy annotaion file 
            annotation_file = "%s/%s" % (def_dataset_path, self.m_nninfo_annotation_file)
            shutil.copy(annotation_file, 
                    self.m_current_code_folder)

            self.gen_tvm_code(tvm_dev_type, tvm_width, tvm_height, tvm_data_type)
            self.make_requirements_file_for_others()
            os.remove(self.get_real_filepath(self.m_nninfo_weight_onnx_file))

        self.m_last_run_state = 0
        # modified
        if self.m_nninfo_yolo_base_file_path != "." and self.m_nninfo_yolo_base_file_path != "" :  
            tmp_folder = self.m_nninfo_yolo_base_file_path.split("/")
            if os.path.exists(self.get_real_filepath(tmp_folder[0])):
                shutil.rmtree(self.get_real_filepath(tmp_folder[0]))
        return 0

    ####################################################################
    @staticmethod
    def load_pt_model():
        """
        Read PyTorch Weight File

        Args: None
        Returns: None
        """

        """
        #remove ".py" substring
        tmp_len = len(self.m_nninfo_class_file)   
        tmp_str = self.m_nninfo_class_file[:(tmp_len-3)]
        #self.m_nninfo_class_name =  'yolo'

        ret = importlib.import_module(tmp_str)
        tmpclass = getattr(ret, self.m_nninfo_class_name)
        self.m_py = tmpclass()

        if self.m_sysinfo_acc_type == 'cpu':
            self.m_pt_model = torch.load(self.m_nnin_weight_file, map_location='cpu')
        else:
            self.m_pt_model = torch.load(self.m_nnin_weight_file, map_location='gpu')
        
        self.m_py.load_state_dict(self.m_pt_model)
        """
        return 0

    ####################################################################
    # onnx 변환
    # 우선은 statix axes먼저, 나중에 dynamic_axes
    @staticmethod
    def conver_to_onnx():
        """
        convert pytorch python-weight file to onnx

        Args: None
        Returns: None
        """
        """
        tmp_val = self.m_nninfo_input_data_type 
        tmp_type = torch.float32
        if tmp_val == 'float64':
            tmp_type = torch.float64
        elif tmp_val == 'float32':
            tmp_type = torch.float32
        elif tmp_val == 'float16':
            tmp_type = torch.float16
        elif tmp_val == 'int64':
            tmp_type = torch.int64
        elif tmp_val == 'int32':
            tmp_type = torch.int32
        elif tmp_val == 'int16':
            tmp_type = torch.int16

        #dummy_data = torch.empty(self.m_nninfo_input_tensor_shape,  dtype = torch.float32)
        dummy_data = torch.empty(self.m_nninfo_input_tensor_shape, dtype=tmp_type)
        torch.onnx.export(self.m_py, dummy_data, self.m_onnx_output_file)
        """
        return 0

    ####################################################################
    # onnx 양자화
    @staticmethod
    def onnx_quantization():
        """
        Quantization for ONNX model

        Args: None
        Returns: None
        """
        return 0

    ####################################################################
    # rknn변환 test
    def gen_rknn_python_code(self):
        """
        Generate python code for RKNN device

        Args: None
        Returns: int
            0 : success
            -1 : error
        """
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        # copy autonn's pytorch to target folder(like prefix folder)
        if self.m_nninfo_yolo_base_file_path != ".":  
            if isinstance(self.m_nninfo_class_file, list):
                m_filelist = self.m_nninfo_class_file
            else:
                m_filelist = [self.m_nninfo_class_file]
            for mfile in m_filelist:
                self.copy_subfolderfile(mfile, self.m_nninfo_yolo_base_file_path)
        # onnx conversion
        # convert .py to onnx and copy
        if isinstance(self.m_nninfo_class_file, list):
            t_file = self.m_nninfo_class_file[0]
        else:
            t_file = self.m_nninfo_class_file
        t_file = "%s%s" % (t_file, "\n")
        tmp = t_file.split("/")
        cnt = len(tmp)
        t_file = tmp[0]
        for  i in range(cnt-1):
            t_file = "%s.%s" % (t_file, tmp[i+1])
        t_file = t_file.split(".py\n")[0]
        t_file = t_file.split("\n")[0]
        t_split = self.m_nninfo_class_name.split("(")
        t_class_name = t_split[0]
        tmp_param = t_split[1].split(")")[0]
        tmp_param = tmp_param.split("=")[1]

        t_list = t_file.split(".")
        t_list_len = len(t_list)
        t_fromlist = ""
        for i in range(t_list_len-1):
            t_fromlist = "%s%s" % (t_fromlist, t_list[i])
            if i < (t_list_len-2): 
                t_fromlist = "%s%s" % (t_fromlist, ".")
        t_impmod = __import__(t_file, fromlist=[t_fromlist])
        t_cls = getattr(t_impmod, t_class_name)

        f_param = self.get_real_filepath(tmp_param[1:-1])
        pt_model = t_cls(cfg=f_param)
        pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
        pt_model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')), strict=False)
        pt_model.eval()
        # convert and save onnx file 
        # the followings must be uncomment, commented out just for testing 
        # khlee
        '''
        dummy = torch.randn(
                self.m_nninfo_input_tensor_shape[0], 
                self.m_nninfo_input_tensor_shape[1], 
                self.m_nninfo_input_tensor_shape[2], 
                self.m_nninfo_input_tensor_shape[3], 
                requires_grad=True)
        torch.onnx.export(pt_model, dummy,
                self.get_real_filepath(self.m_nninfo_weight_onnx_file),
                opset_version = 11,
                export_params = True, # default True  
                # khlee
                # the following  line makes RKNN builder make 'unsupported" Expand --> build time error
                # if the following line is deleted, ONNXruntimeError ShapeInferenceError Incompatible dimensions  -> rknn load time  error
                do_constant_folding = False, # default=True 
                # training=False, aten=False, export_raw_ir=False, operator_export_type=None, enable_onnx_checker=True,
                # verbose=True,
                input_names=['input'],
                output_names=['output'])
        '''
        # khlee only for test copy yolov.onnx to tango.onnx
        # must be deleted 
        o_file = "./db/yolov7.onnx"
        shutil.copy(o_file, self.get_real_filepath(self.m_nninfo_weight_onnx_file))

        # tmp_om = onnx.load(self.get_real_filepath(self.m_nninfo_weight_onnx_file))
        # onnx_out = tmp_om.graph.output
        # o_outputs = [] 
        # for i in range(len(onnx_out)):
        #     tmp = onnx_out[i].name
        #     print(tmp)
        #     o_outputs.append(tmp) 

        self.m_converted_file = "%s%s" % (self.m_current_file_path, def_rknn_file)
        # comment out for test only -> uncomment needed
        # Create RKNN object
        # rknn = RKNN(verbose=True)
        
        # pre-process config
        logging.debug('--> Config model')
        # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
        #            output_tensor_type='int8')

        # Load ONNX model
        # ret = rknn.load_onnx(self.get_real_filepath(self.m_nninfo_weight_onnx_file), outputs=['output'])
        if ret != 0:
            logging.debug('Load model failed!')
            return -1
        logging.debug('--> Loading model')

        # Build model
        # ret = rknn.build(do_quantization=True)
        if ret != 0:
            logging.debug('Build model failed!')
            return -1
        logging.debug('--> Building model')

        # Export RKNN model
        # ret = rknn.export_rknn(self.m_converted_file)
        if ret != 0:
            return -1
        logging.debug('--> Export rknn model')

        # convert  and copy .pt file to nn_model folder  
        if os.path.isfile(self.m_converted_file):
            shutil.copy(self.m_converted_file, self.m_current_code_folder)
            # after converting and copying, remove temporary converted file 
            os.remove(self.m_converted_file)

        # rknn code generation
        tmp_str = "%s%s%s%s" % ("", "import ", self.m_sysinfo_vision_lib, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "import numpy an np", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "from rknnlite.api import RKNNLite", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "import time", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s%s" % (tmp_str, "rknn_model_file = ", def_rknn_file, def_newline)
        # for test change the file name
        tmp_str = "%s%s%s" % (tmp_str, "input_file = '480.mp4'", def_newline)
        tmp_str = "%s%s%s%s" % (tmp_str, def_newline, def_newline, def_newline)

        tmp_str = "%s%s%s%s" % (tmp_str, "IMG_SIZE = ", self.m_nninfo_input_tensor_shape[2],
                                def_newline)
        tmp_str = "%s%s%s%s" % (tmp_str, "BOX_THESH =  ", self.m_nninfo_postproc_iou_thres,
                                def_newline)
        tmp_str = "%s%s%s%s" % (tmp_str, "NMS_THRESH =", self.m_nninfo_postproc_conf_thres,
                                def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s" % (tmp_str, "CLASSES = (")
        for cnt in range(0, self.m_nninfo_number_of_labels):
            if (cnt % 5) == 0:
                tmp_str = "%s%s%s%s" % (tmp_str, def_newline, def_4blank, def_4blank)
            if cnt == (self.m_nninfo_number_of_labels - 1):
                tmp_str = "%s%s%s%s" % (tmp_str, "'", self.m_nninfo_labelmap_info[cnt], "'")
            else:
                tmp_str = "%s%s%s%s%s" % (tmp_str, "'", self.m_nninfo_labelmap_info[cnt],
                                          "'", ', ')
        tmp_str = "%s%s%s" % (tmp_str, ")", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, 'def sigmoid(x):', def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    return 1 / (1 + np.exp(-x))', def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "def xywh2xyxy(x):", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Convert [x, y, w, h] to [x1, y1, x2, y2]",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    y = np.copy(x)", def_newline)
        tmp_str = '%s%s%s' % (tmp_str, '    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x',
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y',
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              '    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x',
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              '    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y',
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    return y', def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "def process(input, mask, anchors):", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    anchors = [anchors[i] for i in mask]",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    grid_h, grid_w = map(int, input.shape[0:2])",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_confidence = sigmoid(input[..., 4])",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    box_confidence = np.expand_dims(box_confidence, axis=-1)",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_class_probs = sigmoid(input[..., 5:])",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_xy = sigmoid(input[..., :2]) * 2 - 0.5",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    grid = np.concatenate((col, row), axis=-1)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_xy += grid", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_xy *= int(IMG_SIZE / grid_h)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_wh = pow(sigmoid(input[..., 2:4]) * 2, 2)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box_wh = box_wh * anchors", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    box = np.concatenate((box_xy, box_wh), axis=-1)",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    return box, box_confidence, box_class_probs",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str,
                              "def filter_boxes(boxes, box_confidences, box_class_probs):",
                              def_newline)
        tmp_str = "%s%s%s%s" % (tmp_str,
                                '    """Filter boxes with box threshold. It\'s a bit different with ',
                                'origin yolov5 post process!',
                                def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Arguments", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        boxes: ndarray, boxes of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        box_confidences: ndarray, confidences of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        box_class_probs: ndarray, class_probs of objects.",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Returns", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        boxes: ndarray, filtered boxes.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        classes: ndarray, classes for boxes.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        scores: ndarray, scores for boxes.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    """', def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    box_classes = np.argmax(box_class_probs, axis=-1)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    box_class_scores = np.max(box_class_probs, axis=-1)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    pos = np.where(box_confidences[..., 0] >= BOX_THESH)",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    boxes = boxes[pos]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    classes = box_classes[pos]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    scores = box_class_scores[pos]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    return boxes, classes, scores", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "def nms_boxes(boxes, scores):", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    """Suppress non-maximal boxes.', def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Arguments", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        boxes: ndarray, boxes of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        scores: ndarray, scores of objects.",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Returns", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        keep: ndarray, index of effective boxes.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    """', def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    x = boxes[:, 0]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    y = boxes[:, 1]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    w = boxes[:, 2] - boxes[:, 0]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    h = boxes[:, 3] - boxes[:, 1]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    areas = w * h", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    order = scores.argsort()[::-1]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    keep = []", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    while order.size > 0:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        i = order[0]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        keep.append(i)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        xx1 = np.maximum(x[i], x[order[1:]])",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        yy1 = np.maximum(y[i], y[order[1:]])",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        inter = w1 * h1", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        ovr = inter / (areas[i] + areas[order[1:]] - inter)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        inds = np.where(ovr <= NMS_THRESH)[0]",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        order = order[inds + 1]",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    keep = np.array(keep)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    return keep", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "def yolov5_post_process(input_data):",
                              def_newline)
        length = len(self.m_nninfo_mask)
        tmp_str = "%s%s" % (tmp_str, "    masks = [")
        for i in range(0, length):
            if (i % 3) == 0:
                tmp_str = "%s%s%s%s" % (tmp_str, def_newline, def_4blank, def_4blank)
            if i == (length - 1):
                tmp_str = "%s%s" % (tmp_str, self.m_nninfo_mask[i])
            else:
                tmp_str = "%s%s%s" % (tmp_str, self.m_nninfo_mask[i], ",")
        tmp_str = "%s%s%s" % (tmp_str, "]", def_newline)
        tmp_str = "%s%s" % (tmp_str, "    anchors = [")
        length = len(self.m_nninfo_anchors)
        for i in range(0, length):
            if (i % 3) == 0:
                tmp_str = "%s%s%s%s" % (tmp_str, def_newline, def_4blank, def_4blank)
            if i == (length - 1):
                tmp_str = "%s%s" % (tmp_str, self.m_nninfo_anchors[i])
            else:
                tmp_str = "%s%s%s" % (tmp_str, self.m_nninfo_anchors[i], ",")
        tmp_str = "%s%s%s" % (tmp_str, "]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "    boxes, classes, scores = [], [], []",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    for input, mask in zip(input_data, masks):",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        b, c, s = process(input, mask, anchors)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        b, c, s = filter_boxes(b, c, s)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        boxes.append(b)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        classes.append(c)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        scores.append(s)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    boxes = np.concatenate(boxes)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    boxes = xywh2xyxy(boxes)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    classes = np.concatenate(classes)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    scores = np.concatenate(scores)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    nboxes, nclasses, nscores = [], [], []",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    for c in set(classes):", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        inds = np.where(classes == c)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        b = boxes[inds]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        c = classes[inds]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        s = scores[inds]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        keep = nms_boxes(b, s)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        nboxes.append(b[keep])", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        nclasses.append(c[keep])", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        nscores.append(s[keep])", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    if not nclasses and not nscores:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        return None, None, None", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    boxes = np.concatenate(nboxes)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    classes = np.concatenate(nclasses)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    scores = np.concatenate(nscores)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    return boxes, classes, scores", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "def draw(image, boxes, scores, classes):",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    """Draw the boxes on the image.', def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # Argument:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        image: original image.", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        boxes: ndarray, boxes of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        classes: ndarray, classes of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        scores: ndarray, scores of objects.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        all_classes: all classes name.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '    """', def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    for box, score, cl in zip(boxes, scores, classes):",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        top, left, right, bottom = box",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        top = int(top)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        left = int(left)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        right = int(right)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        bottom = int(bottom)", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "                    (top, left - 6),", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "                    cv2.FONT_HERSHEY_SIMPLEX,",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "                    0.6, (0, 0, 255), 2)",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "if __name__ == '__main__':", def_newline)
        # parsing parameter
        tmp_str = "%s%s%s" % (tmp_str, "    if len(sys.argv) < 2:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        print('%s%s' % ('input_file = ', input_file))", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    else:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        input_file = sys.argv[1]", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    rknn_lite = RKNNLite()", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    # load RKNN model", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    print('--> Load RKNN model')", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ret = rknn_lite.load_rknn(rknn_model_file)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    if ret != 0:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        print('Load RKNN model failed')",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        exit(ret)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    print('done')", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "    # run on RK356x/RK3588 with Debian OS, do not need specify target.",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ret = rknn_lite.init_runtime()", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    if ret != 0:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        print('Init runtime environment failed')",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        exit(ret)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    print('done')", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ##############################", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ### yolov5 inference START ###", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ##############################", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    cap = cv2.VideoCapture(input_file)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    while True:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        check, frame = cap.read()", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        # Set inputs", def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        # Inference", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        start_time = time.time()", def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        outputs = rknn_lite.inference(inputs=[frame])",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        end_time = time.time()", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "        # post process", def_newline)
        cnt = self.m_nninfo_output_number
        for i in range(0, cnt):
            tmp_str = "%s%s%s%s%s%s%s" % (tmp_str,
                                          "        input", i, "_data = ouputs[", i, "]", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        # yaml의 output_size_format에 따라 수정 필요, 
        # 출력단의 name에 따라 출력 포맷이 달라짐 확인 요망
        for i in range(0, cnt):
            tmp_str = "%s%s%s%s%s%s%s%s%s" % (tmp_str,
                                              "        input", i, "_data = input", i,
                                              "_data.reshape([3, -1]+list(input", i,
                                              "_data.shape[-2:]))", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str, "        input_data = list()", def_newline)

        for i in range(0, cnt):
            tmp_str = "%s%s%s%s%s" % (tmp_str,
                                      "        input_data.append(np.transpose(input",
                                      i, "_data, (1, 2, 0, 3)))", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)

        tmp_str = "%s%s%s" % (tmp_str,
                              "        boxes, classes, scores = post.yolov5_post_process(input_data)",
                              def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        if boxes is not None:", def_newline)
        tmp_str = "%s%s%s" % (tmp_str,
                              "            post.draw(result, boxes, scores, classes)",
                              def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        # show output", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        fps = 1 / (end_time - start_time)",
                              def_newline)
        tmp_str = "%s%s%s%s" % (tmp_str,
                                '        cv2.putText(result, f"{fps:.3f} FPS", (20, 35),',
                                'cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)', def_newline)
        tmp_str = "%s%s%s" % (tmp_str, '        cv2.imshow("result", result)', def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        ret = cv2.waitKey(1)", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "        if (ret >= 0):", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "            break;", def_newline)
        tmp_str = "%s%s" % (tmp_str, def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    cap.release()", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    cv2.destroyAllWindows()", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ############################", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ### yolov5 inference END ###", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    ############################", def_newline)
        tmp_str = "%s%s%s" % (tmp_str, "    rknn_lite.release()", def_newline)

        try:
            outf = open(self.get_code_filepath(self.m_deploy_python_file), "w")
        except IOError as err:
            logging.debug("RKNN Deploy File Write Error #1")
            return -1
        outf.write(tmp_str)
        outf.close()

        # manual copy
        if os.path.isfile(def_m1_manual):
            shutil.copy(def_m1_manual, self.m_current_code_folder)

        # khlee
        # onnx file remove
        os.remove(self.get_real_filepath(self.m_nninfo_weight_onnx_file))
        return 0

    ####################################################################
    def gen_python_code(self):
        """
        convert pytorch template code

        Args: None
        Returns: int
            0 : success
            -1 : error
        """
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        if self.m_nninfo_weight_pt_file == "":
            self.m_converted_file = "%s%s%s" % (self.m_current_file_path, "/", "yolo.pt")
        else:
            self.m_converted_file = "%s%s%s" % (self.m_current_file_path, "/", self.m_nninfo_weight_pt_file)

        if self.m_deploy_type == 'cloud':
            # if sys.version_info.major == 3 and sys.version_info.minor > 7:
            #    shutil.copytree('./db/yolov3/', self.m_current_code_folder, dirs_exist_ok=True)
            # else:
            #     if os.path.exists('./db/yolov3/'):
            #         for file in os.scandir('./db/yolov3/'):
            #            if os.path.isfile(file.path):
            #                shutil.copy(file.path, self.m_current_code_folder)
            #            else:
            #                tname = "%s/%s" % (self.m_current_code_folder, file.name)
            #                shutil.copytree(file.path, tname)
            # zip db/yolov3.db into nn_model foler
            zipfile.ZipFile('./db/yolov3.db').extractall(self.m_current_code_folder)
            # copy db/yolov3/yolov3.pt into nn_model folder
            shutil.copy('./db/yolov3/yolov3.pt', self.m_current_code_folder)
        elif self.m_deploy_type == 'k8s':
            # khlee copy k8s app codes
            zipfile.ZipFile('./db/k8syolov5.db').extractall(self.m_current_code_folder)
            # copy requirement file to code_folder just for testing
            # khlee
            if os.path.isfile(self.get_real_filepath(self.m_requirement_file)):
                shutil.copy(self.get_real_filepath(self.m_requirement_file), self.m_current_code_folder)

        else:
            '''
            # convert  and copy .pt file to nn_model folder
            if os.path.isfile(self.m_converted_file):
                shutil.copy(self.m_converted_file, self.m_current_code_folder)

            tmp_str = "%s%s%s%s%s%s" % ('\"\"\"', def_newline, self.m_deploy_python_file,
                                        def_newline, '\"\"\"', def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "import torch", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "import cv2", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "import numpy as np", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "import time", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, def_newline, def_newline)

            # pytorch file
            # 가속기 gpu, npu 고려 사항 입력
            tmp_str = "%s%s%s" % (tmp_str,
                                  "model = torch.hub.load('ultralytics/yolov5', 'yolov5s')", def_newline)

            # input method
            tmp_str = "%s%s%s" % (tmp_str, "cap = cv2.VideoCapture('480.mp4')", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, def_newline, def_newline)

            tmp_str = "%s%s%s" % (tmp_str, "while True:", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "ret, img = cap.read()", def_newline)
            tmp_str = "%s%s" % (tmp_str, def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "# Inference", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "start_time = time.time()", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "results = model(img)", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "end_time = time.time()", def_newline)
            tmp_str = "%s%s" % (tmp_str, def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank, "fps = 1 / (end_time - start_time)",
                                    def_newline)
            tmp_str = "%s%s%s" % (tmp_str,
                                  def_4blank, 'cv2.putText(img, f"{fps:.3f} FPS", (20,35),')
            tmp_str = "%s%s%s" % (tmp_str,
                                  " cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str, def_4blank,
                                    "cv2.imshow('YoloV5 Test...', np.squeeze(results.render()))", def_newline)
            tmp_str = "%s%s%s%s" % (tmp_str,
                                    def_4blank, "if cv2.waitKey(1) & 0xFF == ord('q'):", def_newline)
            tmp_str = "%s%s%s%s%s" % (tmp_str, def_4blank, def_4blank, "break", def_newline)
            tmp_str = "%s%s" % (tmp_str, def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "cap.release()", def_newline)
            tmp_str = "%s%s%s" % (tmp_str, "cv2.destroyAllWindows()", def_newline)

            try:
                f = open(self.get_code_filepath(self.m_deploy_python_file), 'w')
            except IOError as err:
                print("Python File Write Error", err)
                return -1
            f.write(tmp_str)
            f.close()
            '''
            # khlee
            # copy pytorch related file recursively (make subdirectory and copy files)
            if isinstance(self.m_nninfo_class_file, list):
                num_files = len(self.m_nninfo_class_file)
                if num_files >= 1:
                    for i in range(num_files):
                        self.copy_subfolderfile(self.m_nninfo_class_file[i], self.m_current_code_folder, base_dir=self.m_nninfo_yolo_base_file_path) 
            else:
                shutil.copy(self.get_real_filepath(self.m_nninfo_class_file), self.m_current_code_folder, base_dir=self.m_nninfo_yolo_base_file_path) 
            # copy .pt file
            pt_path = "%s%s" % (self.m_current_file_path, self.m_nninfo_weight_pt_file)
            shutil.copy(pt_path, self.m_current_code_folder)
            # copy basemodel.yaml file
            t_split = self.m_nninfo_class_name.split("(")
            t_class_name = t_split[0]
            tmp_param = t_split[1].split(")")[0]
            tmp_param = tmp_param.split("=")[1]
            f_param = self.get_real_filepath(tmp_param[1:-1])
            shutil.copy(f_param, self.m_current_code_folder)
            # copy myutil file
            shutil.copy(def_trt_myutil_file_name, self.m_current_code_folder)
            try:
                f = open(self.get_code_filepath(self.m_deploy_python_file), 'w')
            except IOError as err:
                logging.debug("Python File Write Error")
                return -1
            
            # copy initial values
            if isinstance(self.m_nninfo_class_file, list):
                t_file = self.m_nninfo_class_file[0]
            else:
                t_file = self.m_nninfo_class_file
            t_file = "%s%s" % (t_file, "\n")
            tmp = t_file.split("/")
            cnt = len(tmp)
            t_file = tmp[0]
            for  i in range(cnt-1):
                t_file = "%s.%s" % (t_file, tmp[i+1])
            t_file = t_file.split(".py\n")[0]
            t_file = t_file.split("\n")[0]
            if self.m_nninfo_yolo_base_file_path != ".":  
                #kkkhhhllleee
                tmp_path = self.m_nninfo_yolo_base_file_path.replace("/", ".")
                t_file = "%s.%s" % (tmp_path, t_file)
            f.write('import %s as ye\n' % t_file)
            f.write('def_pt_file = "%s"\n' % self.m_nninfo_weight_pt_file)
            # remove prefix coco/coco.yaml -> coco.yaml
            a_file = self.m_nninfo_annotation_file.split("/")
            f.write('def_label_yaml = "%s"\n' % a_file[-1])
            if type(self.m_sysinfo_input_method) is str:
                f.write('def_input_location = "%s"\n' % self.m_sysinfo_input_method)
            else:
                f.write('def_input_location = %d\n' % self.m_sysinfo_input_method)
            f.write('def_conf_thres = %s\n' % self.m_nninfo_postproc_conf_thres)
            f.write('def_iou_thres = %s\n'% self.m_nninfo_postproc_iou_thres)
            if type(self.m_sysinfo_output_method) is str:
                f.write('def_output_location = "%s"\n' % self.m_sysinfo_output_method)
            else:
                f.write('def_output_location = %d\n' % self.m_sysinfo_output_method)
            if self.m_sysinfo_acc_type == "cuda":
                f.write('use_cuda = True\n')
            else:
                f.write('use_cuda = False\n')
            f.write('def_data_type = "%s"\n' % self.m_nninfo_input_data_type)
            f.write('def_width = %s\n' % self.m_nninfo_input_tensor_shape[2])
            f.write('def_height = %s\n\n\n' % self.m_nninfo_input_tensor_shape[3])

            # copy head
            try:
                f1 = open("./db/pytorch_template.head", 'r')
            except IOError as err:
                logging.debug("pytorch temp. head open error")
                return -1
            for line1 in f1:
                f.write(line1)
            f1.close()

            # copy model initializing code
            f.write('\n        self.model = ye.Model(cfg="basemodel.yaml")\n')

            # copy body
            try:
                f2 = open("./db/pytorch_template.body", 'r')
            except IOError as err:
                logging.debug("pytorch temp. body open error")
                return -1
            for line2 in f2:
                f.write(line2)
            f2.close()
            # close output.py
            f.close()

        # self.m_nninfo_annotation_file
        annotation_file = "%s/%s" % (def_dataset_path, self.m_nninfo_annotation_file)
        if os.path.isfile(annotation_file):
            shutil.copy(annotation_file, self.m_current_code_folder)

        return 0

    ####################################################################
    def gen_acl_code(self):
        """
        Generate template code for ARM ACL

        Args: None
        Returns: None
        """

        # convert onnx model to tflite model yolo_v3_tiny_fp32.tflite from jaebok park
        # self.m_converted_file = "%s%s%s" % (self.m_current_file_path, "/", "yolo_v3_tiny_darknet_fp32.tflite")
        self.m_converted_file = "./db/yolo_v3_tiny_darknet_fp32.tflite"
        # convert nn model and copy it to nn_model folder 
        # khlee
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


    def gen_tensorrt_code(self, width, height, data_type):
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        # write tensorrt_converter code from DB
        tmpstr = ""
        tmpstr = "%s%s%s" % (tmpstr, "import os", def_newline)
        tmpstr = "%s%s%s" % (tmpstr, "import numpy as np", def_newline)
        tmpstr = "%s%s%s" % (tmpstr, "import tensorrt as trt", def_newline)
        tmpstr = "%s%s%s" % (tmpstr, def_newline, def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_onnx_model = ", '"', 
                self.m_nninfo_weight_onnx_file, '"', def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_calib_cache = ",'"',  
                def_trt_calib_cache, '"', def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_trt_engine = ", '"', 
                def_trt_engine, '"', def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_trt_precision = ", '"',  
                def_trt_precision, '"', def_newline)
        tmpstr = "%s%s%f%s" % (tmpstr, "def_trt_conf_thres = ", 
                self.m_nninfo_postproc_conf_thres, def_newline)
        tmpstr = "%s%s%f%s" % (tmpstr, "def_trt_iou_thres = ", 
                self.m_nninfo_postproc_iou_thres, def_newline)
        tmpstr = "%s%s%s" % (tmpstr, "def_trt_max_detection = 100", def_newline)
        tmpstr = "%s%s%s" % (tmpstr, def_newline, def_newline)

        try:
            conv_outf = open(self.get_code_filepath(def_trt_converter_file_name), "w")
        except IOError as err:
            logging.debug("TensorRT Converter File Write Error #1")
            return -1
        conv_outf.write(tmpstr)
        try:
            ft = open("./db/tensorrt-converter.py", 'r')
        except IOError as err:
            logging.debug("TensorRT converter  body open error")
            return -1
        # body copy
        for line2 in ft:
            conv_outf.write(line2)
        ft.close()
        conv_outf.close()

        # write tensorrt_infer code from DB
        tmpstr = ""
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_model = ", '"', 
                def_trt_engine, '"', def_newline)
        # tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_label_yaml = ", '"',  
        #         self.m_nninfo_annotation_file, '"', def_newline)
        a_file = self.m_nninfo_annotation_file.split("/")
        b_file = a_file[-1]
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_label_yaml = ", '"',  
                b_file, '"', def_newline)
        if type(self.m_sysinfo_input_method) is str:
            tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_input_location = ", '"', 
                    self.m_sysinfo_input_method, '"', def_newline)
        else:
            tmpstr = "%s%s%s%s" % (tmpstr, "def_input_location = ",  
                    self.m_sysinfo_input_method, def_newline)
        if type(self.m_sysinfo_output_method) is str:
            tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_output_location = ", '"', 
                    self.m_sysinfo_output_method, '"', def_newline)  
        else:
            tmpstr = "%s%s%s%s" % (tmpstr, "def_output_location = ",  
                    self.m_sysinfo_output_method, def_newline)  
        tmpstr = "%s%s%s%s" % (tmpstr, "def_conf_thres = ",  
                self.m_nninfo_postproc_conf_thres, def_newline)
        tmpstr = "%s%s%s%s" % (tmpstr, "def_iou_thres = ",  
                self.m_nninfo_postproc_iou_thres, def_newline)
        try:
            infer_outf = open(self.get_code_filepath(def_deploy_python_file), "w")
        except IOError as err:
            logging.debug("TensorRT inference File Write Error #1")
            return -1
        infer_outf.write(tmpstr)
        try:
            fi = open("./db/tensorrt-infer-template.py", 'r')
        except IOError as err:
            logging.debug("TensorRT converter  body open error")
            return -1
        # body copy
        for line2 in fi:
            infer_outf.write(line2)
        fi.close()
        infer_outf.close()
        #copy util file
        shutil.copy(def_trt_myutil_file_name, self.m_current_code_folder)
        #copy calib file
        shutil.copy(def_trt_calib_cache, self.m_current_code_folder)
        # copy annotation file
        annotation_file = "%s/%s" % (def_dataset_path, self.m_nninfo_annotation_file)
        if os.path.isfile(annotation_file):
            shutil.copy(annotation_file, self.m_current_code_folder)
        '''
        # copy model file and related files
        if isinstance(self.m_nninfo_class_file, list):
            num_files = len(self.m_nninfo_class_file)
            if num_files >= 1:
                for i in range(num_files):
                    self.copy_subfolderfile(self.m_nninfo_class_file[i], self.m_current_code_folder)
        else:
            shutil.copy(self.get_real_filepath(self.m_nninfo_class_file), self.m_current_code_folder)
        '''
        return


    def gen_tvm_code(self, dev_type, width, height, data_type):
        if not os.path.exists(self.m_current_code_folder):
            os.makedirs(self.m_current_code_folder)
        tmpstr = ""
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_lib_path = ", '"', 
            def_TVM_lib_path, '"', def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_code_path = ", '"', 
            def_TVM_code_path, '"', def_newline)
        a_file = self.m_nninfo_annotation_file.split("/")
        b_file = a_file[-1]
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_label_yaml = ",'"',  
            b_file, '"', def_newline)
        if type(self.m_sysinfo_input_method) is str:
            tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_input_location = ", '"',   
                    self.m_sysinfo_input_method, '"', def_newline)
        else:
            tmpstr = "%s%s%s%s" % (tmpstr, "def_input_location = ",   
                    self.m_sysinfo_input_method, def_newline)
        if type(self.m_sysinfo_output_method) is str:
            tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_output_location = ", '"',  
                    self.m_sysinfo_output_method, '"', def_newline)  
        else:
            tmpstr = "%s%s%s%s" % (tmpstr, "def_output_location = ",  
                    self.m_sysinfo_output_method, def_newline)  
        tmpstr = "%s%s%s%s" % (tmpstr, "def_conf_thres = ",  
                self.m_nninfo_postproc_conf_thres, def_newline)
        tmpstr = "%s%s%s%s" % (tmpstr, "def_iou_thres = ",  
                self.m_nninfo_postproc_iou_thres, def_newline)
        if self.m_sysinfo_acc_type == "cuda": 
            tmpstr = "%s%s%s%s" % (tmpstr, "def_dev_type = ",  "1", def_newline)
        else:
            tmpstr = "%s%s%s%s" % (tmpstr, "def_dev_type = ",  "0", def_newline)
        tmpstr = "%s%s%s%s%s%s" % (tmpstr, "def_data_type = ", '"',  
                self.m_nninfo_input_data_type, '"', def_newline)
        tmpstr = "%s%s%s%s" % (tmpstr, "def_width = ",  
                self.m_nninfo_input_tensor_shape[1], def_newline)
        tmpstr = "%s%s%s%s" % (tmpstr, "def_height = ",  
                self.m_nninfo_input_tensor_shape[2], def_newline) 
        tmpstr = "%s%s%s" % (tmpstr, def_newline, def_newline) 
        try:
            infer_outf = open(self.get_code_filepath(def_deploy_python_file), "w")
        except IOError as err:
            logging.debug("TVM inference File Write Error #1")
            return -1
        infer_outf.write(tmpstr)
        try:
            fi = open("./db/tvm-infer-template.py", 'r')
        except IOError as err:
            logging.debug("TVM infer  body open error")
            return -1
        # body copy
        for line2 in fi:
            infer_outf.write(line2)
        fi.close()
        infer_outf.close()
        #copy util file
        shutil.copy(def_TVM_myutil_file_name, self.m_current_code_folder)
        # copy annotation file
        annotation_file = "%s/%s" % (def_dataset_path, self.m_nninfo_annotation_file)
        if os.path.isfile(annotation_file):
            shutil.copy(annotation_file, self.m_current_code_folder)
        '''
        # copy model file and related files
        if isinstance(self.m_nninfo_class_file, list):
            num_files = len(self.m_nninfo_class_file)
            if num_files >= 1:
                for i in range(num_files):
                    self.copy_subfolderfile(self.m_nninfo_class_file[i], self.m_current_code_folder)
        else:
            shutil.copy(self.get_real_filepath(self.m_nninfo_class_file), self.m_current_code_folder)
        '''

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

    # Must be modified for tensorrt and tvm 
    # as well as web server application
    ####################################################################
    def make_requirements_file_for_docker(self):  # entry point , work dir
        """
        make yaml file for deployment - docker based environment

        Args: None
        Returns: int
            0 : success
            -1 : error
        """

        if self.m_deploy_type == 'cloud' or self.m_deploy_type == 'k8s': 
            # khlee 
            # self.m_deploy_type == 'PCServer': just for webserver not docker
            dep_type = 'docker'
        else:
            dep_type = 'native'

        # if self.m_sysinfo_engine_type == 'rknn':
        #     w_filw = def_rknn_file
        # elif self.m_sysinfo_engine_type == 'acl':
        #     w_filw = "yolo_v3_tiny_darknet_fp32.tflite"
        # #  .pt file
        # else:
        #    w_filw = self.m_nninfo_weight_pt_file

        my_entry = ['python3', 'deploy_server.py']
        # my_entry = self.m_deploy_entrypoint

        if self.m_deploy_type == 'cloud':
            if self.m_sysinfo_engine_type == 'pytorch':
                req_pkg = self.m_nninfo_yolo_require_packages
            else:
                req_pkg = []
            for item in self.m_nninfo_user_libs:
                if not item in req_pkg:
                    req_pkg.append(item)
            req_pkg.append('vim')
            t_pkg = {"atp": req_pkg, "pypi": ['flask==1.2.3']}
            t_com = {"custom_packages": t_pkg,
                     "libs": ['python==3.9', 'torch>=1.1.0'], 
                     "engine": self.m_sysinfo_engine_type # added
                    }
            t_build = {'architecture': 'linux/amd64',
                       "accelerator": 'cpu',
                       "os": 'ubuntu',
                       # "target_name": 'yolov3:latest',
                       "components": t_com }
            t_deploy = {"type": dep_type,
                        "workdir": "/test/test",
                        "entrypoint": my_entry,
                        "network": {"service_host_ip": self.m_deploy_network_hostip,
                                    "service_host_port": self.m_deploy_network_hostport,
                                    "service_container_port": self.m_deploy_network_serviceport}}
            if self.m_sysinfo_engine_type == 'tensorrt':
                t_deploy['pre_exec'] = ['tensorrt-converter.py']
            t_total = {"build": t_build, 
                    "deploy": t_deploy}
        elif self.m_deploy_type == "k8s":
            if self.m_sysinfo_engine_type == 'pytorch':
                req_pkg = self.m_nninfo_yolo_require_packages
            else:
                req_pkg = []
            for item in self.m_nninfo_user_libs:
                if not item in req_pkg:
                    req_pkg.append(item)
            req_pkg.append('vim')
            req_pkg.append('python3.9')
            t_pkg = {"atp": req_pkg, "pypi": []}
            t_com = {"custom_packages": t_pkg,
                     "engine": 'pytorch' }
            t_build = {'architecture': 'linux/amd64',
                       "accelerator": 'gpu',
                       "os": 'ubuntu',
                       "target_name": 'ultralytics/yolo5:v6.1',
                       "components": t_com
                       }
            t_deploy = {"type": dep_type,
                     "workdir": "/test/test",
                     'entrypoint': [ '/bin/bash', '-c'], # my_entry,
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
            if self.m_sysinfo_engine_type == 'tensorrt':
                t_deploy['pre_exec'] = ['tensorrt-converter.py']
            t_opt = {"nn_file": 'detect.py',
                     "weight_file": self.m_nninfo_weight_pt_file,
                     "annotation_file": 'coco128.yaml'}
            t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}
        else:
            t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
            t_com = {"engine": "pytorch",
                     "libs": ['python==3.9', 'torch>=1.1.0'],
                     "custom_packages": t_pkg}
            t_build = {'architecture': self.m_sysinfo_cpu_type,
                       "accelerator": self.m_sysinfo_acc_type,
                       "os": self.m_sysinfo_os_type, 
                       "components": t_com,
                       #"engine": self.m_sysinfo_engine_type, 
                       "target_name": "yolov3:latest"}
            t_deploy = {"type": dep_type, "work_dir": self.m_deploy_work_dir,
                        "workdir": self.m_deploy_work_dir,
                        "entrypoint": my_entry,
                        "network": {"service_host_ip": self.m_deploy_network_hostip,
                                    "service_host_port": self.m_deploy_network_hostport,
                                    "service_container_port": self.m_deploy_network_serviceport}}
            if self.m_sysinfo_engine_type == 'tensorrt':
                t_deploy['pre_exec'] = ['tensorrt-converter.py']
            t_opt = {"nn_file": 'yolo.py',
                     "weight_file": self.m_nninfo_weight_pt_file,
                     "annotation_file": self.m_nninfo_deploy_annotation_file}

            t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}

        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        # khlee 
        # copy deployment.yaml to m_current_filepath
        shutil.copy(r_file, self.m_current_file_path)
        return 0
        
        
    ####################################################################
    def make_requirements_file_for_PCServer(self):
        """
        make yaml file for deployment  - for PC webserver

        Args: None
        Returns: int
            0 : success
            -1 : error
        """ 
        if self.m_sysinfo_engine_type == 'pytorch':
            dev_req = "%s/%s" % (self.m_current_code_folder, def_ondevice_python_requirements)
            if len(self.m_nninfo_yolo_require_packages) > 0 or len(self.m_nninfo_user_libs) > 0:
                with open(dev_req, "w")  as f:
                    for item in self.m_nninfo_yolo_require_packages:
                        f.write(item)
                        f.write("\n")
                    for item in self.m_nninfo_user_libs:
                        f.write(item)
                        f.write("\n")

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
        if self.m_sysinfo_engine_type == 'tensorrt':
            t_deploy['pre_exec'] = ['tensorrt-converter.py']
        t_opt = {"nn_file": self.m_deploy_python_file,
                 "weight_file": w_filw,
                 "annotation_file": self.m_nninfo_annotation_file}
        t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}

        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        # khlee 
        # copy deployment.yaml to m_current_filepath
        shutil.copy(r_file, self.m_current_file_path)
        return
        



    ####################################################################
    def make_requirements_file_for_others(self):
        """
        make yaml file for deployment  - stand-alone PC or ondevices

        Args: None
        Returns: int
            0 : success
            -1 : error
        """ 
        logging.debug(self.m_sysinfo_engine_type) 
        if self.m_sysinfo_engine_type == 'pytorch':
            dev_req = "%s/%s" % (self.m_current_code_folder, def_ondevice_python_requirements)
            if len(self.m_nninfo_yolo_require_packages) > 0 or len(self.m_nninfo_user_libs) > 0:
                with open(dev_req, "w")  as f:
                    for item in self.m_nninfo_yolo_require_packages:
                        f.write(item)
                        f.write("\n")
                    for item in self.m_nninfo_user_libs:
                        f.write(item)
                        f.write("\n")

        t_pkg = {"atp": self.m_sysinfo_apt, "pypi": self.m_sysinfo_papi}
        if self.m_sysinfo_engine_type == 'rknn':
            w_filw = def_rknn_file
            t_com = {"engine": "rknn", "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        elif self.m_sysinfo_engine_type == 'acl':
            w_filw = "yolo_v3_tiny_darknet_fp32.tflite"
            t_com = {"engine": "acl", "libs": self.m_sysinfo_libs,
                 "custom_packages": t_pkg}
        elif self.m_sysinfo_engine_type == 'tensorrt':
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
        if self.m_sysinfo_engine_type == 'tensorrt':
            t_deploy['pre_exec'] = ['tensorrt-converter.py']
        t_opt = {"nn_file": self.m_deploy_python_file,
                 "weight_file": w_filw,
                 "annotation_file": self.m_nninfo_annotation_file}
        t_total = {"build": t_build, "deploy": t_deploy, "optional": t_opt}

        try:
            r_file = "%s/%s" % (self.m_current_code_folder, self.m_requirement_file)
            f = open(r_file, 'w')
        except IOError as err:
            logging.debug("Yaml File for deployment write error")
            return -1
        yaml.dump(t_total, f)
        f.close()
        # khlee 
        # copy deployment.yaml to m_current_filepath
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
                    elif self.m_flag == 1:
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
