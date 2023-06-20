import sys
import tvm
import onnx
import tvm.relay as relay

# model_path = "./onnx/yolov7-tiny_640x640.onnx"        
#ok model_path = "./onnx/yolov7-tiny_640x640.onnx"        
#error model_path = "./onnx/yolov7_post_256x320.onnx"
#error model_path = "./onnx/yolov7-tiny_post_256x320.onnx"   
#error model_path = "./onnx/yolov7_post_256x480.onnx"
#error model_path = "./onnx/yolov7-tiny_post_256x480.onnx"   
#error model_path = "./onnx/yolov7_post_256x640.onnx"
#error model_path = "./onnx/yolov7-tiny_post_256x640.onnx"   
#error model_path = "./onnx/yolov7_post_384x640.onnx"
#error model_path = "./onnx/yolov7-tiny_post_384x640.onnx"   
#error model_path = "./onnx/yolov7_post_480x640.onnx"
#error model_path = "./onnx/yolov7-tiny_post_480x640.onnx"   
model_path = "./onnx/yolov7_post_640x640.onnx"
#error model_path = "./onnx/yolov7-tiny_post_640x640.onnx"   


# 0 for x86, 1 for cuda, 2 for arm, 3 for opencl
def_TVM_dev_type = 0 
def_TVM_width = 224 
def_TVM_height = 224 
def_TVM_lib_path = "mylib.so"
def_TVM_code_path = "mycode.bin"
def_TVM_model_path = "nn_model.onnx"

class TVMConverter:
    dev_type = def_TVM_dev_type
    data_type = def_TVM_data_type
    width = def_TVM_width  
    height = def_TVM_height  

    def run(self, dev_type=def_TVM_dev_type, 
            model_path=def_TVM_model_path, 
            lib_path=def_TVM_lib_path,
            code_path=def_TVM_code_path):
        onnx_model = onnx.load(model_path)
        input_name = onnx_model.graph.input[0].name
        tensor_type = onnx_model.graph.input[0].type.tensor_type
        tmp_list = []
        if (tensor_type.HasField("shape")):
            for d in tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    tmp_list.append(d.dim_value)
            i_shape = tuple(tmp_list)
            (x_, y_, self.width, self.height) = i_shape 
            print(i_shape)
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
        print(dtype)

        if dev_type == 0:
            target = "llvm"
        elif dev_type == 1:
            target = "cuda"
        elif dev_type == 2:
            target = "llvm -mtriple=aarch64-linux-gnu"
        else:
            target = "opencl"
        print(target)

        shape_dict = {input_name: i_shape}
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        executable = relay.vm.compile(mod, target=target, params=params)
        code, lib = executable.save()
        lib.export_library(lib_path)
        with open(code_path, "wb") as outf:
            outf.write(code)

if __name__=='__main__':
    tvm_converter = TVMConverter()
    tvm_converter.run(dev_type=def_TVM_dev_type, 
            model_path=def_TVM_model_path, 
            lib_path=def_TVM_lib_path,
            code_path=def_TVM_code_path)

##########################################################
#compiled_graph_lib = relay.build_module.build(mod, target=target, params=params)
#compiled_graph_lib.export_library(save_name)
# vmc = relay.backend.vm.VMCompiler()
#with tvm.autotvm.apply_graph_best("test.log"):
# vm = vmc.compile(mod, target=target, params=params) 
#code, lib = vm.save()
#lib.export_library("mylib.so")
#with open("mycode.bin", "wb") as outf:
    #outf.write(code)


