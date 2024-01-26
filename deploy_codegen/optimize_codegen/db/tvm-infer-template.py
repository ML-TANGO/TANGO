'''
def_mod_path = "yolov7.mod"
def_param_path = "yolov7.param"
def_label_yaml = "coco.yaml"
def_conf_thres = 0.4
def_iou_thres = 0.4
def_dev_type = "llvm"  # 0=tvm.cpu(), 1=tvm.cuda(), other=tvm.opencl()
def_data_type = "float32"
def_width = 640 
def_height = 640
def_input_location = "./images" # number=camera, url, or file_path
def_output_location = "./result" # 0=screen, 1=text, url,or folder_path
'''

# from user's selection
# input_location can be  a number = camera device id(defualt 0) or
#                        url = stream input or
#                        directory or
#                        file name
# output_location can be number 0 = screen drawing or
#                        number 1 = text output or
#                        url = stream output or
#                        directory 


""" copyright notice
This module is for testing code for neural network model.
"""
###############################################################
import onnx
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
import matplotlib.pyplot as plt

#############################################
# Class definition for TVM run module  
#############################################
class TVMRun():
    def __init__(self,  
            dev_type=def_dev_type, lyaml=def_label_yaml, 
            input_location=def_input_location, confthr=def_conf_thres, 
            iouthr=def_iou_thres, output_location=def_output_location):
        """
        TVM Runtime class definition
        Args:
            lyaml : yaml file for label info. 
            input_location : [0-9]: camera, URL, file name, or folder name 
            confthr : confidence threshhold 
            iouthr : IOU threshhole
            output_location : 0=screen, 1=text output, or folder name 
        """
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
        self.dtype = "float32"
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
        '''
        self.onnx_model = onnx.load(self.lib_path)
        input_name = "images"
        self.shape_dict = {input_name: [1, 3, self.width, self.height]}
        mod, params = tvm.relay.frontend.from_onnx(self.onnx_model, self.shape_dict)
        with open("aaa.mod", "w") as fo:
            fo.write(tvm.ir.save_json(mod))
        with open("aaa.param", "wb") as fo:
            fo.write(tvm.runtime.save_param_dict(params))
        '''
        with open(def_mod_path, "r") as fi:
            mod = tvm.ir.load_json(fi.read())
        with open(def_param_path, "rb") as fi:
            params = tvm.runtime.load_param_dict(fi.read())
        with tvm.transform.PassContext(opt_level=1):
            self.executor = tvm.relay.build_module.create_executor(
            "graph", mod, tvm.cpu(0), self.dev_type, params
            ).evaluate()


    def iou(self, box1: list, box2: list):
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        if area_box1 <= 0 or area_box2 <= 0:
            iou_value = 0
        else:
            y_min_intersection = max(box1[1], box2[1])
            x_min_intersection = max(box1[0], box2[0])
            y_max_intersection = min(box1[3], box2[3])
            x_max_intersection = min(box1[2], box2[2])

            area_intersection = max(0, y_max_intersection - y_min_intersection) * \
                                max(0, x_max_intersection - x_min_intersection)
            area_union = area_box1 + area_box2 - area_intersection
    
            try:
                iou_value = area_intersection / area_union
            except ZeroDivisionError:
                iou_value = 0
    
        return iou_value

    def nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep


    def multiclass_nms(self, boxes, scores, nms_thr=0.45, score_thr=0.1):
        final_dets = []
        num_classes = scores.shape[1]
        for cls_ind in range(num_classes):
            cls_scores = scores[:, cls_ind]
            valid_score_mask = cls_scores > score_thr
            if valid_score_mask.sum() == 0:
                continue
            else:
                valid_scores = cls_scores[valid_score_mask]
                valid_boxes = boxes[valid_score_mask]
                keep = self.nms(valid_boxes, valid_scores, nms_thr)
                if len(keep) > 0:
                    cls_inds = np.ones((len(keep), 1)) * cls_ind
                    dets = np.concatenate(
                        [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                    )
                    final_dets.append(dets)
        if len(final_dets) == 0:
            return None
        return np.concatenate(final_dets, 0)
    

    def yolo_processing(self, predictions):
        confidence_det = predictions[:, :, 4][0]
        detections = list(np.where(confidence_det > self.conf_thres)[0])
        t_box = []
        t_score = []
        all_det = []
        for d in detections:
            t_box.append(predictions[:, d, :4][0])
            t_score.append(predictions[:, d, 4][0] * predictions[:, d, 5:][0])
        boxes = np.array(t_box)
        scores = np.array(t_score)
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= self.ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=self.iou_thres, score_thr=self.conf_thres)
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        result = [final_boxes, final_scores, final_cls_inds]
        return result

    
    def rainbow_fill(self, size=50):  
        cmap = plt.get_cmap('jet')
        color_list = []
        for n in range(size):
            color = cmap(n/size)
            color_list.append(color[:3])  
        return np.array(color_list)


    def preprocess(self, image):
        swap = (2, 0, 1)
        padded_img = np.ones((self.width, self.height, 3)) * 114.0
        img = np.array(image)
        r = min(self.width / img.shape[0], self.height / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def postprocess(self, image, pred, save_path, still_image=False):
        scores = pred[1]
        boxes = pred[0]
        cls_ids = pred[2]
        _COLORS = self.rainbow_fill(80).astype(np.float32).reshape(-1, 3)
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < self.conf_thres:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])
            color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(self.classes[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                image,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(image, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)
    
        if self.view_img:
            cv2.imshow("result", image)
            cv2.waitKey(1)
            time.sleep(1)
        elif self.save_img:
            if still_image:
                cv2.imwrite(str(save_path), image)
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
            # preprocessing
            x, self.ratio = self.preprocess(img)
            #run
            tvm_output = self.executor(tvm.nd.array(x.astype(self.dtype))).numpy()
            # interpreter ret value
            ret =  self.yolo_processing(tvm_output)
            self.postprocess(img, ret, save_path)
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
            # preprocessing
            x, self.ratio = self.preprocess(img)
            #run
            tvm_output = self.executor(tvm.nd.array(x.astype(self.dtype))).numpy()
            # interpreter ret value
            ret =  self.yolo_processing(tvm_output)
            self.postprocess(img, ret, save_path)
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
            # preprocessing
            x, self.ratio = self.preprocess(img)
            #run
            tvm_output = self.executor(tvm.nd.array(x.astype(self.dtype))).numpy()
            # interpreter ret value
            ret =  self.yolo_processing(tvm_output)
            self.postprocess(img, ret, save_path)
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
        # preprocessing
        x, self.ratio = self.preprocess(img)
        #run
        tvm_output = self.executor(tvm.nd.array(x.astype(self.dtype))).numpy()
        # interpreter ret value
        ret =  self.yolo_processing(tvm_output)
        self.postprocess(img, ret, save_path, still_image=True)
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




if __name__ == "__main__":
    mytvm = TVMRun( 
            dev_type=def_dev_type,
            input_location= def_input_location, 
            confthr=def_conf_thres, 
            iouthr=def_iou_thres, 
            output_location= def_output_location
            )
    mytvm.load_model()
    mytvm.run()
