'''
import os
import numpy as np
import tensorrt as trt

#def_onnx_model = "newonnx/yolov7-d6.onnx"  
#def_onnx_model = "newonnx/yolov7-e6e.onnx"   
#def_onnx_model = "newonnx/yolov7-w6.onnx"  
#def_onnx_model = "newonnx/yolov7x.onnx"
#def_onnx_model = "newonnx/yolov7-e6.onnx"  
#def_onnx_model = "newonnx/yolov7-tiny.onnx"  
def_onnx_model = "newonnx/yolov7.onnx"
def_calib_cache = "calibration.cache"
def_trt_engine = "v7-16.trt"
def_trt_precision = "fp16"
# def_trt_precision = "int8"
def_trt_conf_thres = 0.4
def_trt_iou_thres = 0.5
def_trt_max_detection = 100
'''


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

        '''
        print("hehehe")
        previous_output = self.network.get_output(0)
        print(previous_output.shape)
        self.network.unmark_output(previous_output)
        # output [1, 8400, 85]
        # slice boxes, obj_score, class_scores
        strides = trt.Dims([1,1,1])
        starts = trt.Dims([0,0,0])
        bs, num_boxes, temp = previous_output.shape
        shapes = trt.Dims([bs, num_boxes, 4])
        # [0, 0, 0] [1, 8400, 4] [1, 1, 1]
        boxes = self.network.add_slice(previous_output, 
                starts, shapes, strides)
        num_classes = temp -5 
        starts[2] = 4
        shapes[2] = 1
        # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
        obj_score = self.network.add_slice(previous_output, 
                starts, shapes, strides)
        starts[2] = 5
        shapes[2] = num_classes
        # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
        scores = self.network.add_slice(previous_output, 
                starts, shapes, strides)
        # scores = obj_score * class_scores => [bs, num_boxes, nc]
        scores = self.network.add_elementwise(obj_score.get_output(0), 
                scores.get_output(0), trt.ElementWiseOperation.PROD)
        registry = trt.get_plugin_registry()
        creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
        fc = []
        fc.append(trt.PluginField("background_class", 
            np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("max_output_boxes", 
            np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("score_threshold", 
            np.array([conf_thres], dtype=np.float32), 
            trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("iou_threshold", 
            np.array([iou_thres], dtype=np.float32), 
            trt.PluginFieldType.FLOAT32))
        fc.append(trt.PluginField("box_coding", 
            np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
        fc.append(trt.PluginField("score_activation", 
            np.array([0], dtype=np.int32), trt.PluginFieldType.INT32))
        
        fc = trt.PluginFieldCollection(fc) 
        nms_layer = creator.create_plugin("nms_layer", fc)

        layer = self.network.add_plugin_v2([boxes.get_output(0), 
            scores.get_output(0)], nms_layer)
        layer.get_output(0).name = "num"
        layer.get_output(1).name = "boxes"
        layer.get_output(2).name = "scores"
        layer.get_output(3).name = "classes"
        for i in range(4):
            self.network.mark_output(layer.get_output(i))
        '''


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
            trt_conf_thres=def_trt_conf_thres, 
            trt_iou_thres=def_trt_iou_thres, 
            trt_max_detection=def_trt_max_detection, 
            trt_engine=def_trt_engine, 
            trt_precision=def_trt_precision):
        builder.create_network(onnx_model, trt_conf_thres, 
            trt_iou_thres, trt_max_detection)
        builder.create_engine(trt_engine, trt_precision)
        return

if __name__ == "__main__":
    builder = Build_engine()
    builder.run()
