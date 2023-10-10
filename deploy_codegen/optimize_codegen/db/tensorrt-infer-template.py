'''
def_trt_engine = "v7-16.trt"
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
def_output_location = 0 # "./result" # 0=screen, 1=text, url,or folder_path
def_conf_thres = 0.5
def_iou_thres = 0.4
def_calib_cache = "calibration.cache"
def_trt_engine = "v7-16.trt"
def_trt_precision = "fp16"
def_trt_max_detection = 100
'''

""" copyright notice
This module is for testing code for neural network model.
"""
###########################################################
###########################################################
import cv2
import numpy as np
import time
import yaml
import os
import sys
import myutil
import matplotlib.pyplot as plt

# for inference engine
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


###############################################################
class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """
    Implements the INT8 Entropy Calibrator 2.
    """

    def __init__(self, cache_file):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None


    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.
        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

class Build_engine:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 
                8 << 30)
        self.network = None
        self.parser = None

    def create_network(self, onnx_model, conf_thres, iou_thres, max_det):
        self.network = self.builder.create_network(1 << 
                int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self.trt_logger)
        onnx_model = os.path.realpath(onnx_model)
        with open(onnx_model, "rb") as f:
            self.parser.parse(f.read())
            
    def create_engine(self, engine_path, precision):
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)

        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                print("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                print("INT8 is not supported natively on this platform/device")
            else:
                if self.builder.platform_has_fast_fp16:
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(def_calib_cache)

        with self.builder.build_serialized_network(self.network, self.config) as engine, open(engine_path, "wb") as f:
            print("Serializing engine to file: {:}".format(engine_path))
            f.write(engine)  # .serialize()

    def run(self, onnx_model= def_onnx_model, 
            trt_conf_thres= def_conf_thres,
            trt_iou_thres=def_iou_thres, 
            trt_max_detection=def_trt_max_detection, 
            trt_engine=def_trt_engine, 
            trt_precision=def_trt_precision):
        self.create_network(onnx_model, trt_conf_thres, 
            trt_iou_thres, trt_max_detection)
        print("TensorRT model is being generated!! It takes time!!")
        self.create_engine(trt_engine, trt_precision)
        print("TensorRT engine is created !!!")
            
            
            


#############################################
# Class definition for TensorRT run module  
#############################################
class TRTRun():
    def __init__(self, model_path=def_trt_engine, lyaml=def_label_yaml, 
            input_location=def_input_location, confthr=def_conf_thres, 
            iouthr=def_iou_thres, output_location=def_output_location):
        """
        TensorRT Runtime class definition
        Args:
            model_path : tensorRT model path 
            lyaml : yaml file for label info. 
            input_location : [0-9]: camera, URL, file name, or folder name 
            confthr : confidence threshhold 
            iouthr : IOU threshhole
            output_location : 0=screen, 1=text output, or folder name 
        """
        self.model_path = model_path
        self.bindings = None
        self.engine = None
        self.context = None
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
        self.width = 640
        self.height = 480
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
        # transform onnx to trt
        self.builder = Build_engine()
        self.builder.run()
        return

    def load_model(self):
        """
        To load tensorrt model 

        Args:
            none
        Returns: 
            none
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER) 
        trt.init_libnvinfer_plugins(None, "") 
        with open(self.model_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.imgsz = self.engine.get_binding_shape(0)[2:]  
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        return

    def preprocess(self, image):
        """
        To preprocess image for neural network input 

        Args:
            org_img : image data 
        Returns: 
            preprocessed image data
        """
        swap=(2, 0, 1)
        if len(image.shape) == 3:
            padded_img = np.ones((self.imgsz[0], self.imgsz[1], 3)) * 114.0
        else:
            padded_img = np.ones(self.imgsz) * 114.0
        img = np.array(image)
        r = min(self.imgsz[0] / img.shape[0], self.imgsz[1] / img.shape[1])
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
            preproc_image, ratio  = self.preprocess(img)
            # inference
            self.inputs[0]['host'] = np.ravel(preproc_image)
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # inference
            self.context.execute_async_v2(
                bindings = self.bindings,
                stream_handle = self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            data = [out['host'] for out in self.outputs]

            predictions = np.reshape(data, (1, -1, int(5+len(self.classes))))[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result = [final_boxes, final_scores, final_cls_inds]
            self.postprocess(img, result, save_path)
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
            preproc_image, ratio  = self.preprocess(img)
            # inference
            self.inputs[0]['host'] = np.ravel(preproc_image)
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # inference
            self.context.execute_async_v2(
                bindings = self.bindings,
                stream_handle = self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            data = [out['host'] for out in self.outputs]

            predictions = np.reshape(data, (1, -1, int(5+len(self.classes))))[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result = [final_boxes, final_scores, final_cls_inds]
            self.postprocess(img, result, save_path)
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
            preproc_image, ratio  = self.preprocess(img)
            # inference
            self.inputs[0]['host'] = np.ravel(preproc_image)
            for inp in self.inputs:
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            # inference
            self.context.execute_async_v2(
                bindings = self.bindings,
                stream_handle = self.stream.handle)
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            data = [out['host'] for out in self.outputs]

            predictions = np.reshape(data, (1, -1, int(5+len(self.classes))))[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4:5] * predictions[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
            boxes_xyxy /= ratio
            dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            result = [final_boxes, final_scores, final_cls_inds]
            self.postprocess(img, result, save_path)
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
        preproc_image, ratio  = self.preprocess(img)
        # inference
        self.inputs[0]['host'] = np.ravel(preproc_image)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # inference
        self.context.execute_async_v2(
            bindings = self.bindings,
            stream_handle = self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]

        predictions = np.reshape(data, (1, -1, int(5+len(self.classes))))[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = self.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        result = [final_boxes, final_scores, final_cls_inds]
        self.postprocess(img, result, save_path, still_image=True)
        return

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

    def rainbow_fill(self, size=50):  # simpler way to generate rainbow color
        cmap = plt.get_cmap('jet')
        color_list = []
        for n in range(size):
            color = cmap(n/size)
            color_list.append(color[:3])  # might need rounding? (round(x, 3) for x in color)[:3]
        return np.array(color_list)

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
        boxes, scores, cls_ids = label
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


if __name__ == "__main__":
    mytrt = TRTRun(model_path=def_trt_engine, 
            input_location= def_input_location, 
            confthr=def_conf_thres, 
            iouthr=def_iou_thres, 
            output_location= def_output_location
            )
    mytrt.load_model()
    mytrt.run()
