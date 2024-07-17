import logging
import argparse
import contextlib
import json
import yaml
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from tango.common.models.experimental import attempt_load, End2End
from tango.common.models.yolo import    (   Detect,
                                            IDetect,
                                            IKeypoint,
                                            IAuxDetect,
                                            IBin,
                                            Model,
                                        )
# from tango.utils.dataloaders import LoadImages
from tango.utils.general import (   check_dataset,
                                    check_img_size,
                                    check_requirements,
                                    colorstr,
                                )
from tango.utils.torch_utils import select_device

logger = logging.getLogger(__name__)

def export_formats():
    # YOLO export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['ONNX END2END', 'onnx_end2end', '_end2end.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ['TensorFlow.js', 'tfjs', '_web_model', False, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def export_torchscript(model, im, file, optimize, prefix=colorstr('TorchScript:')):
    # YOLO TorchScript model export
    logger.info(f'{prefix} starting export with torch {torch.__version__}...')
    try:
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        logger.info('TorchScript export success, saved as %s' % f)
        return f, ts
    except Exception as e:
        logger.warn('TorchScript export failure: %s' % e)
        return None, None


def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr('ONNX:')):
    # YOLO ONNX export
    # check_requirements('onnx')
    import onnx
    logger.info(f'{prefix} starting export with onnx {onnx.__version__}...')

    try:
        f = file.with_suffix('.onnx')

        # output_names = ['output0', 'output1'] if isinstance(model, SegmentationModel) else ['output0']
        output_names = ['output0']
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
            # if isinstance(model, SegmentationModel):
            #     dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            #     dynamic['output1'] = {0: 'batch', 2: 'mask_height', 3: 'mask_width'}  # shape(1,32,160,160)
            # elif isinstance(model, DetectionModel):
            #     dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}  # shape(1,25200,85)

        torch.onnx.export(
            model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic else im,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        d = {'stride': int(max(model.stride)), 'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        # onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                cuda = torch.cuda.is_available()
                # check_requirements(('onnxruntime-gpu' if cuda else 'onnxruntime', 'onnx-simplifier>=0.4.1'))
                import onnxsim

                logger.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                # onnx.save(model_onnx, f)
                logger.info(f'{prefix} simplifier success')
            except Exception as e:
                logger.warn(f'{prefix} simplifier failure: {e}')

        onnx.save(model_onnx, f)
        logger.info('ONNX export success, saved as %s' % f)
        return f, model_onnx
    except Exception as e:
        logger.warn('ONNX export failure: %s' % e)
        return None, None


def export_onnx_end2end(model,
                        im,
                        file,
                        simplify,
                        topk_all,
                        iou_thres,
                        conf_thres,
                        device,
                        labels,
                        prefix=colorstr('ONNX END2END:')):
    # YOLO ONNX export
    # check_requirements('onnx')
    import onnx
    logger.info(f'{prefix} starting export with onnx {onnx.__version__}...')

    try:
        f = os.path.splitext(file)[0] + "-end2end.onnx"
        batch_size = 'batch'

        dynamic_axes = {'images': {0 : 'batch', 2: 'height', 3:'width'}, } # variable length axes

        output_axes = {
                        'num_dets': {0: 'batch'},
                        'det_boxes': {0: 'batch'},
                        'det_scores': {0: 'batch'},
                        'det_classes': {0: 'batch'},
                    }
        dynamic_axes.update(output_axes)
        model = End2End(model, topk_all, iou_thres, conf_thres, None ,device, labels)

        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        shapes = [ batch_size, 1,  batch_size,  topk_all, 4,
                   batch_size,  topk_all,  batch_size,  topk_all]

        torch.onnx.export(  model,
                            im,
                            f,
                            verbose=False,
                            export_params=True,       # store the trained parameter weights inside the model file
                            opset_version=12,
                            do_constant_folding=True, # whether to execute constant folding for optimization
                            input_names=['images'],
                            output_names=output_names,
                            dynamic_axes=dynamic_axes)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        for i in model_onnx.graph.output:
            for j in i.type.tensor_type.shape.dim:
                j.dim_param = str(shapes.pop(0))

        if simplify:
            try:
                import onnxsim

                logger.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                logger.info(f'{prefix} simplifier success')
            except Exception as e:
                logger.warn(f'{prefix} simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(model_onnx,f)
        logger.info('ONNX END2END export success, saved as %s' % f)
        return f, model_onnx
    except Exception as e:
        logger.warn('ONNX END2END export failure: %s' % e)
        return None, None


def export_openvino(file, metadata, half, prefix=colorstr('OpenVINO:')):
    # YOLO OpenVINO export
    check_requirements('openvino-dev')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
    import openvino.inference_engine as ie

    logger.info(f'\n{prefix} starting export with openvino {ie.__version__}...')
    f = str(file).replace('.pt', f'_openvino_model{os.sep}')

    #cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} --data_type {'FP16' if half else 'FP32'}"
    #cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} {"--compress_to_fp16" if half else ""}"
    half_arg = "--compress_to_fp16" if half else ""
    cmd = f"mo --input_model {file.with_suffix('.onnx')} --output_dir {f} {half_arg}"
    subprocess.run(cmd.split(), check=True, env=os.environ)  # export
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)  # add metadata.yaml
    return f, None


def export_tensorrt(model,
                    im,
                    file,
                    half,
                    dynamic,
                    simplify,
                    workspace=4,
                    verbose=False,
                    prefix=colorstr('TensorRT:')):
    # YOLO TensorRT export https://developer.nvidia.com/tensorrt
    # assert im.device.type != 'cpu', 'export running on CPU but must be on GPU, i.e. `python export.py --device 0`'
    onnx, _ = export_onnx(model, im, file, 12, dynamic, simplify)

    try:
        import tensorrt as trt
    except Exception as e:
        logger.warn(f'{prefix} export failure: {e}')
        return None, None

    import pkg_resources as pkg
    current, minimum, maximum = (pkg.parse_version(x) for x in (trt.__version__, '8.0.0', '10.1.0'))
    if minimum >= current or current >= maximum:
        logger.warn(f'{prefix} export failure: {prefix}>={minimum},<={maximum} required by autonn in ml-tango')
        return None, None

    is_trt10 = int(trt.__version__.split(".")[0]) >= 10

    # onnx = file.with_suffix('.onnx')

    logger.info(f'{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx.exists(), f'failed to export ONNX file: {onnx}'
    try:
        f = file.with_suffix('.engine')  # TensorRT engine file
        logger_trt = trt.Logger(trt.Logger.INFO)
        if verbose:
            logger_trt.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger_trt)
        config = builder.create_builder_config()
        workspace = int(workspace * (1 << 30))
        if is_trt10:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        else:
            config.max_workspace_size = workspace

        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger_trt)
        if not parser.parse_from_file(str(onnx)):
            raise RuntimeError(f'failed to load ONNX file: {onnx}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            logger.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            logger.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        if dynamic:
            if im.shape[0] <= 1:
                logger.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
            profile = builder.create_optimization_profile()
            for inp in inputs:
                profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
            config.add_optimization_profile(profile)

        logger.info(f'{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
        if builder.platform_has_fast_fp16 and half:
            config.set_flag(trt.BuilderFlag.FP16)

        if is_trt10:
            build = builder.build_serialized_network # trt version >=10.0,0
        else:
            build = builder.build_engine # trt version <10.0.0

        with build(network, config) as engine, open(f, 'wb') as t:
            t.write(engine if is_trt10 else engine.serialize())

        logger.info('TensorRT export success, saved as %s' % f)
        return f, None
    except Exception as e:
        logger.warn('TensorRT export failure: %s' % e)
        return None, None


def export_tf_saved_model(model,
                          im,
                          file,
                          dynamic,
                          tf_nms=False,
                          agnostic_nms=False,
                          topk_per_class=100,
                          topk_all=100,
                          iou_thres=0.45,
                          conf_thres=0.25,
                          keras=False,
                          prefix=colorstr('TensorFlow SavedModel:')):
    # YOLO TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        # check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
        # import tensorflow as tf
        logger.warn(f'{prefix} tensorflow import failure')
        return None, None
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from tango.common.models.tf import TFModel

    logger.info(f'{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(tfm,
                            f,
                            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if check_version(
                                tf.__version__, '2.6') else tf.saved_model.SaveOptions())
    return f, keras_model


def export_tf_pb(keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLO TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    logger.info(f'{prefix} starting export with tensorflow {tf.__version__}...')
    f = file.with_suffix('.pb')

    m = tf.function(lambda x: keras_model(x))  # full model
    m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
    frozen_func = convert_variables_to_constants_v2(m)
    frozen_func.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
    return f, None


def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    import tensorflow as tf

    logger.info(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace('.pt', '-fp16.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        # from models.tf import representative_dataset_gen
        # dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=imgsz, auto=False)
        # converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        f = str(file).replace('.pt', '-int8.tflite')
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None


def export_edgetpu(file, prefix=colorstr('Edge TPU:')):
    # YOLO Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    cmd = 'edgetpu_compiler --version'
    help_url = 'https://coral.ai/docs/edgetpu/compiler/'
    assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
    if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
        logger.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
        sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
        for c in (
                'curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -',
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list',
                'sudo apt-get update', 'sudo apt-get install edgetpu-compiler'):
            subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
    ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

    logger.info(f'\n{prefix} starting export with Edge TPU compiler {ver}...')
    f = str(file).replace('.pt', '-int8_edgetpu.tflite')  # Edge TPU model
    f_tfl = str(file).replace('.pt', '-int8.tflite')  # TFLite model

    cmd = f"edgetpu_compiler -s -d -k 10 --out_dir {file.parent} {f_tfl}"
    subprocess.run(cmd.split(), check=True)
    return f, None


def export_tf_js(file, prefix=colorstr('TensorFlow.js:')):
    # YOLO TensorFlow.js export
    check_requirements('tensorflowjs')
    import tensorflowjs as tfjs

    logger.info(f'\n{prefix} starting export with tensorflowjs {tfjs.__version__}...')
    f = str(file).replace('.pt', '_web_model')  # js dir
    f_pb = file.with_suffix('.pb')  # *.pb path
    f_json = f'{f}/model.json'  # *.json path

    cmd = f'tensorflowjs_converter --input_format=tf_frozen_model ' \
          f'--output_node_names=Identity,Identity_1,Identity_2,Identity_3 {f_pb} {f}'
    subprocess.run(cmd.split())

    json = Path(f_json).read_text()
    with open(f_json, 'w') as j:  # sort JSON Identity_* in ascending order
        subst = re.sub(
            r'{"outputs": {"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}, '
            r'"Identity.?.?": {"name": "Identity.?.?"}}}', r'{"outputs": {"Identity": {"name": "Identity"}, '
            r'"Identity_1": {"name": "Identity_1"}, '
            r'"Identity_2": {"name": "Identity_2"}, '
            r'"Identity_3": {"name": "Identity_3"}}}', json)
        j.write(subst)
    return f, None


def add_tflite_metadata(file, metadata, num_outputs):
    # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
    with contextlib.suppress(ImportError):
        # check_requirements('tflite_support')
        from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

        tmp_file = Path('/tmp/meta.txt')
        with open(tmp_file, 'w') as meta_f:
            meta_f.write(str(metadata))

        model_meta = _metadata_fb.ModelMetadataT()
        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        model_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [_metadata_fb.TensorMetadataT()]
        subgraph.outputTensorMetadata = [_metadata_fb.TensorMetadataT()] * num_outputs
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()


def export_config(src, dst, data, base, device, engine):

    nn_dict = {}

    # NN Model
    if engine == 'pytorch':
        weight = 'bestmodel.torchscript'
        config = ''
    else:
        weight = 'bestmodel.onnx'
        config = ''
    nn_dict['weight_file'] = weight
    nn_dict['config_file'] = config

    # Label
    nc = data.get('nc')
    ch = data.get('ch', 3)
    names = data.get('names')
    if not nc and names:
        nc = len(names)
    if not names and nc:
        names = list(range(nc)) # default name: [0, 1, ..., nc-1]
    nn_dict['nc'] = nc
    nn_dict['names'] = names

    # Input
    imgsz = base.get('imgsz', 640)
    input_tensor_shape = [1, ch, imgsz, imgsz]
    device = select_device(device)
    if device.type == 'cpu':
        input_data_type = 'fp32'
    else:
        input_data_type = 'fp16'
    anchors = base.get('anchors')
    if not anchors:
        logger.warn(f'Model Exporter: not found anchor imformation')

    nn_dict['input_tensor_shape'] = input_tensor_shape
    nn_dict['input_data_type'] = input_data_type
    nn_dict['anchors'] = anchors

    # Output
    output_number = 3
    stride = [8, 16, 32]
    output_size = [
        [1, ch, imgsz/stride[0], imgsz/stride[0], 5+nc],
        [1, ch, imgsz/stride[1], imgsz/stride[1], 5+nc],
        [1, ch, imgsz/stride[2], imgsz/stride[2], 5+nc]
    ]
    need_nms = True
    conf_thres = 0.25
    iou_thres = 0.45

    nn_dict['output_number'] = output_number
    nn_dict['output_size'] = output_size
    nn_dict['stride'] = stride
    nn_dict['need_nms'] = need_nms
    nn_dict['conf_thres'] = conf_thres
    nn_dict['iou_thres'] = iou_thres

    with open(dst, 'w') as f:
        yaml.dump(nn_dict, f, default_flow_style=False)


def export_weight(weights, device, include):
    t = time.time()
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, onnx_end2end, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    # file = Path(url2file(weights) if str(weights).startswith(('http:/', 'https:/')) else weights)  # PyTorch weights
    file = Path(weights)

    # options ----------------------------------------------------------------------------------------------------------
    imgsz = [640, 640]  # input image size
    batch_size = 1      # inference batch size
    inplace = True      # YOLO Detect(): set to compute tensors w/o copy
    half = True         # FP16 quantization / GPU only
    int8 = False        # CoreML/TF: INT8 quantization
    dynamic = False     # ONNX/TF/TensorRT: dynamic input sizes
    simplify = True     # ONNX/ONNX-E2E/TensorRT: simplifies the graph for onnx (TensorRT requires ONNX first)
    opset = 12          # ONNX: operator-set version
    workspace = 4.0     # TensorRT: max space size(GB) for tensorrt
    verbose = False     # TensorRT: detail logging for tensorrt
    nms = False         # TF: add nms
    agnostic_nms = True # TF: add nms to tensorflow
    optimize = False    # TorchScript: mobile optimization / CPU only
    keras = False       # TF: save keras model as well
    topk_per_class= 100 # TF.js: nms - top-k per class to keep
    topk_all = 100      # ONNX-E2E/TF.js: nms -  top-k for all classes to keep
    iou_thres = 0.45    # ONNX-E2E/TF.js: nms -  iou threshold
    conf_thres = 0.25   # ONNX-E2E/TF.js: nms -  confidence threshold
    #-------------------------------------------------------------------------------------------------------------------

    # Load PyTorch model
    device = select_device(device)
    if half and device.type == 'cpu':
        logger.warn(f'model exporter: --half only compatible with GPU export, ignore --half')
        half = False
    if half and dynamic:
        logger.warn(f'model exporter: --half not compatible with --dynamic, ignore --dynamic')
        dynamic = False
    model = attempt_load(weights, map_location=device)  # load FP32 model
    logger.debug(model)

    # Checks
    imgsz = [imgsz] if isinstance(imgsz, int) else imgsz
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        # if isinstance(m, (Detect, DDetect, DualDetect, DualDDetect)):
        if isinstance(m, (Detect, IDetect, IKeypoint, IAuxDetect, IBin)):
            m.inplace = inplace
            m.dynamic = dynamic
            if onnx_end2end:
                m.end2end = True
                m.export = False
            else:
                m.end2end = False
                m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16

    # shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)  # model output shape
    shape = []
    if isinstance(y, (tuple, list)):
        for yi in y:
            shape.append(yi.shape)
    else:
        shape.append(y.shape)
    metadata = {'stride': int(max(model.stride)), 'names': model.names}  # model metadata
    # logger.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")
    logger.info(f"{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({os.path.getsize(file) / 1E6:.1f} MB)")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=FutureWarning) # torch.onnx.__patch_torch.__graph_op will be deprecated
    if jit:  # TorchScript
        f[0], ts_model = export_torchscript(model, im, file, optimize)
    if engine:  # TensorRT required ONNX
        f[1], rt_model = export_tensorrt(model, im, file, half, dynamic, simplify, workspace, verbose=verbose)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)
    if onnx_end2end:
        # if isinstance(model, DetectionModel):
        #     labels = model.names
        #     f[2], onnx_model = export_onnx_end2end(model, im, file, simplify, topk_all, iou_thres, conf_thres, device, len(labels))
        # else:
        #     raise RuntimeError("The model is not a DetectionModel.")
        labels = model.names
        f[2], onnx_model = export_onnx_end2end(model, im, file, simplify, topk_all, iou_thres, conf_thres, device, len(labels))
    if xml:  # OpenVINO
        f[3], _ = export_openvino(file, metadata, half)
    if coreml:  # CoreML
        f[4], _ = export_coreml(model, im, file, int8, half)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
        # assert not isinstance(model, ClassificationModel), 'ClassificationModel export to TF formats not yet supported.'
        f[5], s_model = export_tf_saved_model(  model.cpu(),
                                                im,
                                                file,
                                                dynamic,
                                                tf_nms=nms or agnostic_nms or tfjs,
                                                agnostic_nms=agnostic_nms or tfjs,
                                                topk_per_class=topk_per_class,
                                                topk_all=topk_all,
                                                iou_thres=iou_thres,
                                                conf_thres=conf_thres,
                                                keras=keras)
        if pb or tfjs:  # pb prerequisite to tfjs
            f[6], tf_model = export_tf_pb(s_model, file)
        if tflite or edgetpu:
            f[7], tflite_model = export_tflite(s_model, im, file, int8 or edgetpu, data=data, nms=nms, agnostic_nms=agnostic_nms)
            if edgetpu:
                f[8], _ = export_edgetpu(file)
            add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(s_model.outputs))
        if tfjs:
            f[9], _ = export_tf_js(file)
    if paddle:  # PaddlePaddle
        f[10], _ = export_paddle(model, im, file, metadata)

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        logger.info(f'Export complete ({time.time() - t:.1f}s)')
        logger.info(f"Results saved to {colorstr('bold', file.parent.resolve())}")
        logger.info(f"Visualize:       https://netron.app")

    return f  # return list of exported files/dirs

