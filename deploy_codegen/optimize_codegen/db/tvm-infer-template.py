'''
def_lib_path = "mylib.so"
def_code_path = "mycode.bin"
def_label_yaml = "coco.yaml"
# from user's selection
# input_location can be  a number = camera device id(defualt 0) or
#                        url = stream input or
#                        directory or
#                        file name
# output_location can be number 0 = screen drawing or
#                        number 1 = text output or
#                        url = stream output or
#                        directory 
def_input_location = "./images" # number=camera, url, or file_path
def_output_location = "./result" # 0=screen, 1=text, url,or folder_path
def_conf_thres = 0.7
def_iou_thres = 0.5
def_dev_type = 0 # 0=tvm.cpu(), 1=tvm.cuda(), other=tvm.opencl()
def_data_type = "float32"
def_width = 640 
def_height = 640
'''

""" copyright notice
This module is for testing code for neural network model.
"""
###############################################################
import cv2
import numpy as np
import time
import yaml
import os
import sys
import myutil
import tvm
from tvm.runtime import vm as _vm
###
from tvm.relay import transform

#############################################
# Class definition for TVM run module  
#############################################
class TVMRun():
    def __init__(self, lib_path=def_lib_path, code_path=def_code_path, 
            dev_type=def_dev_type, lyaml=def_label_yaml, 
            input_location=def_input_location, confthr=def_conf_thres, 
            iouthr=def_iou_thres, output_location=def_output_location):
        """
        TVM Runtime class definition
        Args:
            lib_path : TVM lib path 
            code_path : TVM code  path 
            lyaml : yaml file for label info. 
            input_location : [0-9]: camera, URL, file name, or folder name 
            confthr : confidence threshhold 
            iouthr : IOU threshhole
            output_location : 0=screen, 1=text output, or folder name 
        """
        self.lib_path = lib_path
        self.code_path = code_path
        self.dev_type = dev_type
        self.data_type = def_data_type
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.classes = None
        self.label_yaml =  lyaml
        self.img_folder = input_location 
        self.conf_thres = confthr 
        self.iou_thres = iouthr 
        self.output_location = output_location 
        self.video = 0
        self.vid_writer = 0
        self.vid_path = ""
        self.text_out = False
        self.view_img = False
        self.save_img = False
        self.stream_out = False
        self.width = def_width
        self.height = def_height
        if self.output_location == 0:
            self.view_img = True
        elif self.output_location == 1:
            self.text_out = True
        elif "://" in self.output_location:
            self.stream_out = True
        else:
            self.save_img = True
        with open(self.label_yaml) as f:
            classes = yaml.safe_load(f)
            self.classes = classes['names']
        return

    def load_model(self):
        """
        To load tensorrt model 

        Args:
            none
        Returns: 
            none
        """
        loaded_lib = tvm.runtime.load_module(self.lib_path)
        loaded_code = bytearray(open(self.code_path, "rb").read())
        des_exec = _vm.Executable.load_exec(loaded_code, loaded_lib)
        if self.dev_type == 0:
            dev = tvm.cpu()
        elif self.dev_type == 1:
            dev = tvm.cuda()
        elif self.dev_type == 2:
            dev = tvm.cuda()
            dev = tvm.cpu()
        else:
            dev = tvm.opencl()
            # dev = tvm.target.opencl()
            # dev = tvm.target.Target("opencl")
        self.vm = _vm.VirtualMachine(des_exec, tvm.cuda())

        # we have to find data type and shape for the input tensor
        # data_type =  
        # input_size = self.inputs[0]['shape'][-2:] 
        # self.width = input_size[1] 
        # self.height = input_size[0]
        return

    def preprocess(self, org_img):
        """
        To preprocess image for neural network input 

        Args:
            org_img : image data 
        Returns: 
            preprocessed image data
        """
        input_size = [100, 100]
        org_h, org_w, org_c = org_img.shape
        image = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        input_size[1] = self.width
        input_size[0] = self.height
        r_w = input_size[1] / org_w
        r_h = input_size[0] / org_h
        if r_h > r_w:
            tx1 = 0
            tx2 = 0
            tw = input_size[1]
            th = int(r_w *  org_h)
            ty1 = int((input_size[0] - th) / 2)
            ty2 = input_size[0] - th - ty1
        else:
            ty1 = 0
            ty2 = 0
            tw = int(r_h * org_w)
            th = input_size[0]
            tx1 = int((input_size[1] - tw) / 2)
            tx2 = input_size[1] - tw - tx1
        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128))
        image = image.astype(np.float32)
        image /= 255.0
        image = np.expand_dims(image, axis=0)
        ret_img = np.ascontiguousarray(image)
        return ret_img

    def do_camera_infer(self, dev=0, target_folder=""):
        """
        To inference from camera input  

        Args:
            dev : camera device number (defualt: 0)
        Returns: 
            none
        """
        save_path = myutil.get_fullpath(self.save_folder, filename)
        self.video = cv2.VideoCapture(dev)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("Press q to quit")
        while (self.video.isOpened()):
            flag, img = self.video.read()
            if flag == False:
                break
            preproc_image = self.preprocess(img)
            # inference
            image = preproc_image.transpose(0, 3, 1, 2) 
            image = np.ascontiguousarray(image)
            cuda.memcpy_htod(self.inputs[0]['allocation'], image)
            self.context.execute_v2(self.allocations)
            for o in range(len(self.outputs)):
                cuda.memcpy_dtoh(self.outputs[o]['host_allocation'],
                    self.outputs[o]['allocation'])
            num_detections = self.outputs[0]['host_allocation']
            nmsed_boxes = self.outputs[1]['host_allocation']
            nmsed_scores = self.outputs[2]['host_allocation']
            nmsed_classes = self.outputs[3]['host_allocation']
            result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
            self.postprocess(preproc_image, result, save_path)
            if cv2.waitKey(1) == ord('q'):
                break
        self.video.release()
        if isinstance(self.vid_writer, cv2.VideoWriter):
            self.vid_writer.release()  
        cv2.destroyAllWindows()
        return

    def do_url_infer(self, url, target_folder=""):
        """
        To inference from streamin data  

        Args:
            url : url for streaming input 
        Returns: 
            none
        """
        save_path = myutil.get_fullpath(target_folder, filename)
        self.video = cv2.VideoCapture(url)
        print("Press q to quit")
        while (self.video.isOpened()):
            flag, img = self.video.read()
            if flag == False:
                break
            preproc_image = self.preprocess(img)
            # inference
            image = preproc_image.transpose(0, 3, 1, 2) 
            image = np.ascontiguousarray(image)
            cuda.memcpy_htod(self.inputs[0]['allocation'], image)
            self.context.execute_v2(self.allocations)
            for o in range(len(self.outputs)):
                cuda.memcpy_dtoh(self.outputs[o]['host_allocation'],
                    self.outputs[o]['allocation'])
            num_detections = self.outputs[0]['host_allocation']
            nmsed_boxes = self.outputs[1]['host_allocation']
            nmsed_scores = self.outputs[2]['host_allocation']
            nmsed_classes = self.outputs[3]['host_allocation']
            result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
            self.postprocess(preproc_image, result, save_path)
            if cv2.waitKey(1) == ord('q'):
                break
        self.video.release()
        if isinstance(self.vid_writer, cv2.VideoWriter):
            self.vid_writer.release()  
        cv2.destroyAllWindows()
        return

    def do_video_infer(self, filename, target_folder=""):
        """
        To inference from video file  

        Args:
            filename : video file name  
        Returns: 
            none
        """
        save_path = myutil.get_fullpath(target_folder, filename)
        if filename == "":
            i_file = self.img_folder
        else:
            i_file = filename
        self.video = cv2.VideoCapture(i_file)
        print("Press q to quit")
        while (self.video.isOpened()):
            flag, img = self.video.read()
            if flag == False:
                break
            preproc_image = self.preprocess(img)
            # inference
            image = preproc_image.transpose(0, 3, 1, 2) 
            image = np.ascontiguousarray(image)
            cuda.memcpy_htod(self.inputs[0]['allocation'], image)
            self.context.execute_v2(self.allocations)
            for o in range(len(self.outputs)):
                cuda.memcpy_dtoh(self.outputs[o]['host_allocation'],
                    self.outputs[o]['allocation'])
            num_detections = self.outputs[0]['host_allocation']
            nmsed_boxes = self.outputs[1]['host_allocation']
            nmsed_scores = self.outputs[2]['host_allocation']
            nmsed_classes = self.outputs[3]['host_allocation']
            result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
            self.postprocess(preproc_image, result, save_path)
            if cv2.waitKey(1) == ord('q'):
                break
        self.video.release()
        if isinstance(self.vid_writer, cv2.VideoWriter):
            self.vid_writer.release()  
        cv2.destroyAllWindows()
        return

    def do_image_infer(self, filename="", target_folder=""):
        """
        To inference from image file  

        Args:
            filename : image file name  
        Returns: 
            none
        """
        save_path = myutil.get_fullpath(target_folder, filename)
        if filename == "":
            i_file = self.img_folder
        else:
            i_file = filename
        print("filename %s, target_folder= %s" % (filename, target_folder))
        print("img file = %s" % i_file)
        img = cv2.imread(i_file)
        if img is None:
            print("input file open error!!!")
            return
        preproc_image = self.preprocess(img)
        # inference
        image = preproc_image.transpose(0, 3, 1, 2) 
        image = np.ascontiguousarray(image)
        #run
        ret = self.vm.run(image)
        #return value check
        # interpreter ret value
        ttt_scores = ret[0].asnumpy().tolist()
        num_detections = len(ttt_scores)
        tmp_scores = [] 
        tmp_boxes = [] 
        tmp_classes = [] 
        # valid_boxes = []
        #for i, score in enumerate(ret[1].asnumpy().tolist()):
        #    valid_boxes.append(boxes[i])
        valid_boxes = ret[1].asnumpy().tolist()
        for i in range(num_detections):
            tmp_scores.append(ttt_scores[i][0])
            tmp_boxes.append(valid_boxes[i][-1:-5:-1])
            tmp_classes.append(valid_boxes[i][1])
        nmsed_scores = [tmp_scores]
        nmsed_boxes = [tmp_boxes] 
        nmsed_classes = [tmp_classes] 
        result = [num_detections, nmsed_boxes, nmsed_scores, nmsed_classes]
        self.postprocess(preproc_image, result, save_path, still_image=True)
        return

    def run(self):
        """
        To call inference fuction  

        Args:
            none
        Returns: 
            none
        """
        f_type = myutil.check_file_type(self.img_folder)
        if self.output_location == 0 or self.output_location == 1:
            self.save_folder = ""
        else:
            self.save_folder = self.output_location 
        if f_type == "camera":
            self.do_camera_infer(self.img_folder, self.save_folder)
        elif f_type == "url":  
            self.do_url_infer(self.img_folder, self.save_folder)
        elif f_type == "video":
            self.do_video_infer(self.img_folder, self.save_folder)
        elif f_type == "image":
            self.do_image_infer(self.img_folder, self.save_folder)
        elif f_type == "directory": 
            for i, filename in enumerate(os.listdir(self.img_folder)):
                full_name = os.path.join(self.img_folder, filename)
                self.do_image_infer(full_name, self.save_folder)
        elif f_type == "unknown":
            print("unkown input!! Halt!!!")
        return

    def postprocess(self, image, label, save_path, still_image=False):
        """
        To postprocessing after inference  

        Args:
            image : original image that was input image for the neural 
                    network model
            label : label infomation 
            save_path : the folder name to save a image that has 
                    object detection info. 
            still_image : whether the original image is read 
                    from a image file or not  
        Returns: 
            none
        """
        num_detections, nmsed_boxes, nmsed_scores, nmsed_classes = label
        # to be modified
        h, w = image.shape[1:3]
        image = np.squeeze(image)
        image *= 255
        image = image.astype(np.uint8)
        for i in range(int(num_detections)):
            if nmsed_scores[0][i] > self.conf_thres:
                detected = str(self.classes[int(
                    nmsed_classes[0][i])]).replace('‘', '').replace('’', '')
                confidence_str = str(nmsed_scores[0][i])
                x1 = int(nmsed_boxes[0][i][0])
                y1 = int(nmsed_boxes[0][i][1])
                x2 = int(nmsed_boxes[0][i][2])
                y2 = int(nmsed_boxes[0][i][3])
                color = (139, 139, 139)
                #color = (81, 81, 81)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), 
                        color, 2)
                text_size, _ = cv2.getTextSize(str(detected), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_w, text_h = text_size
                image = cv2.rectangle(image, (x1, y1-5-text_h), 
                        (x1+text_w, y1), color, -1)
                image = cv2.putText(image, str(detected), 
                        (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 255, 255), 1)
                print("Detect " + str(i+1) + "(" + str(detected) + ")")
                print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, 
                    x2, y2))
                print("Confidence : {:.7f}".format(nmsed_scores[0][i]))

            if self.text_out:
                print("Detect " + str(i+1) + "(" + str(detected) + ")")
                print("Coordinates : [{:d}, {:d}, {:d}, {:d}]".format(x1, y1, 
                    x2, y2))
                print("Confidence : {:.7f}".format(nmsed_scores[0][i]))
        if self.view_img:
            cv2.imshow("result", image)
            cv2.waitKey(1)
            time.sleep(1)
        elif self.save_img:
            if still_image:
                cv2.imwrite(str(save_path), 
                        cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                if self.vid_path != save_path: 
                    self.vid_path = save_path
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  
                    if self.video:  
                        fps = self.video.get(cv2.CAP_PROP_FPS)
                        w, h = image.shape[1], image.shape[0]
                    else:  
                        fps, w, h = 10, image.shape[1], image.shape[0]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.vid_writer = cv2.VideoWriter(save_path, 
                            fourcc, fps, (w, h))
                self.vid_writer.write(image)
        elif self.stream_out:
            # if you want to send detection results to internet
            # write proper codes here to do it
            pass
        return


if __name__ == "__main__":
    mytvm = TVMRun(lib_path=def_lib_path, code_path=def_code_path, 
            dev_type=def_dev_type,
            input_location= def_input_location, 
            confthr=def_conf_thres, 
            iouthr=def_iou_thres, 
            output_location= def_output_location
            )
    mytvm.load_model()
    mytvm.run()
