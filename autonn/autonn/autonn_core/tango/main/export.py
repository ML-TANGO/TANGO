import logging
logger = logging.getLogger(__name__)

import contextlib
import json
import yaml
import os
import platform
import re
import subprocess
import time
from datetime import datetime
import warnings
from pathlib import Path
COMMON_ROOT = Path("/shared/common")
DATASET_ROOT = Path("/shared/datasets")
CORE_DIR = Path(__file__).resolve().parent.parent.parent # /source/autonn_core
CFG_PATH = CORE_DIR / 'tango' / 'common' / 'cfg'
INF_PATH = CORE_DIR / 'tango' / 'inference'

import pandas as pd
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from tango.common.models.experimental import attempt_load, End2End
from tango.common.models.yolo import    (   Detect,
                                            DDetect,
                                            DualDDetect,
                                            IDetect,
                                            IAuxDetect,
                                            Model,
                                        )
# from tango.utils.dataloaders import LoadImages
from tango.utils.general import (   check_dataset,
                                    check_img_size,
                                    check_requirements,
                                    colorstr,
                                )
from tango.utils.torch_utils import select_device




def export_formats():
    # YOLO export formats
    x = [
            ['PyTorch', '-', '.pt', True, True],
            ['TorchScript', 'torchscript', '.torchscript', True, True],
            ['ONNX', 'onnx', '.onnx', True, True],
            ['ONNX END2END', 'onnx_end2end', '-end2end.onnx', True, True],
            ['OpenVINO', 'openvino', '_openvino_model', True, False],
            ['TensorRT', 'engine', '.engine', False, True],
            ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
            ['TensorFlow GraphDef', 'pb', '.pb', True, True],
            ['TensorFlow Lite', 'tflite', '.tflite', True, False],
            ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
        ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def export_torchscript(model, im, file, optimize, task='detection', prefix=colorstr('TorchScript:')):
    # YOLO TorchScript model export
    logger.info(f'\nModel Exporter: {prefix} Starting export with torch {torch.__version__}...')
    try:
        f = file.with_suffix('.torchscript')

        ts = torch.jit.trace(model, im, strict=False)
        if task == 'detection':
            d = {"shape": im.shape, "stride": int(max(model.stride)), "names": model.names}
        elif task == 'classification':
            d = {"shape": im.shape, "names": model.names}
        extra_files = {'config.txt': json.dumps(d)}  # torch._C.ExtraFilesMap()
        if optimize:  # https://pytorch.org/tutorials/recipes/mobile_interpreter.html
            optimize_for_mobile(ts)._save_for_lite_interpreter(str(f), _extra_files=extra_files)
        else:
            ts.save(str(f), _extra_files=extra_files)
        logger.info(f'Model Exporter: {colorstr("TorchScript ")}export success, saved as {f}')
        return f, ts
    except Exception as e:
        logger.warning(f'Model Exporter: {colorstr("TorchScript ")}export failure: {e}')
        return None, None


def export_onnx(model, im, file, opset, dynamic, simplify, task='detection', prefix=colorstr('ONNX:')):
    import onnx
    logger.info(f'\nModel Exporter: {prefix} Starting export with onnx {onnx.__version__}...')

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
            dynamic_axes=dynamic or None
        )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Metadata
        if task == 'detection':
            d = {'stride': int(max(model.stride)), 'names': model.names}
        elif task == 'classification':
            d = {'names': model.names}
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)
        # onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                import onnxsim
                logger.info(f'Model Exporter: {prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                logger.info(f'Model Exporter: {prefix} simplifier success')
                # import onnxslim
                # logger.info(f'Model Exporter: {prefix} slimming with onnxlim {onnxslim.__version__}...')
                # model_onnx = onnxslim.slim(model_onnx)
            except Exception as e:
                logger.warning(f'Model Exporter: {prefix} simplifier failure: {e}')

        onnx.save(model_onnx, f)
        logger.info(f'Model Exporter: {colorstr("ONNX ")}export success, saved as {f}')
        return str(f), model_onnx
    except Exception as e:
        logger.warning(f'Model Exporter: {colorstr("ONNX ")}export failure: {e}')
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
                        v9,
                        prefix=colorstr('ONNX END2END:')):
    # YOLO ONNX export
    # check_requirements('onnx')
    import onnx
    logger.info(f'\nModel Exporter: {prefix} Starting export with onnx {onnx.__version__}...')

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

        model = End2End(model, topk_all, iou_thres, conf_thres, None ,device, labels, v9)

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

                logger.info(f'Model Exporter: {prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                logger.info(f'Model Exporter: {prefix} simplifier success')
            except Exception as e:
                logger.warn(f'Model Exporter: {prefix} simplifier failure: {e}')

        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        onnx.save(model_onnx,f)
        logger.info(f'Model Exporter: {colorstr("ONNX END2END ")}export success, saved as {f}')
        return f, model_onnx
    except Exception as e:
        logger.warning(f'Model Exporter: {colorstr("ONNX END2END ")}export failure: {e}')
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


def export_tf(f_onnx, metadata, int8, imgsz, batch_size, stride=1, prefix=colorstr("TF Saved Model:")):
    # import tensorflow as tf
    import numpy as np
    import onnx2tf
    import shutil

    try:
        logger.info(f"\nModel Exporter: {prefix} Starting export with onnx2tf {onnx2tf.__version__}...")

        f = Path(str(f_onnx).replace('.onnx', '_saved_model'))
        if f.is_dir():
            shutil.rmtree(f)  # delete output folder
        
        onnx2tf_file = Path(INF_PATH / "calibration_image_sample_data_20x128x128x3_float32.npy")
        if not onnx2tf_file.exists():
            logger.warning(f"Model Exporter: {prefix} {onnx2tf_file} does not exist.")
            url = 'github.com/ML-TANGO/TANGO/assets/releases/'
            try:
                torch.hub.download_url_to_file(
                    url, 
                    f'{onnx2tf_file}.zip'
                )
            except Exception as e:
                logger.warning(f'{prefix} Failed to download {onnx2tf_file} from {url}')

        np_data = None
        if int8:
            f.mkdir()
            tmp_file = f / "tmp_tflite_int8_calibration_images.npy"  # int8 calibration images file
            data_yaml = str(DATASET_ROOT / 'coco128/dataset.yaml')
            if os.path.isfile(data_yaml):
                with open(data_yaml) as representative_data:
                    data = yaml.load(representative_data, Loader=yaml.SafeLoader)
            images = []
            dataloader = get_int8_calibration_dataloader(data, imgsz, batch_size, stride)
            for im, _, _, _ in dataloader:
                images.append(im)
            images = torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=imgsz).permute(0, 2, 3, 1)
            np.save(str(tmp_file), images.numpy().astype(np.float32))  # BHWC
            np_data = [["images", tmp_file, [[[[0, 0, 0]]]], [[[[255, 255, 255]]]]]]
        
        keras_model = onnx2tf.convert(
            input_onnx_file_path=f_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity="error",  # note INT8-FP16 activation bug https://github.com/ultralytics/ultralytics/issues/15873
            output_integer_quantized_tflite=int8,
            quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
            custom_input_op_name_np_data_path=np_data,
            disable_group_convolution=True,  # for end-to-end model compatibility
            enable_batchmatmul_unfold=True,  # for end-to-end model compatibility
        )
        with open(f / "metadata.yaml", 'w') as yaml_file:
            yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}, yaml_file, sort_keys=False)
        logger.info(f'Model Exporter: {colorstr("TF Lite: ")}external meta file saved as {f / "metadata.yaml"}.')

        # Remove/rename TFLite model
        if int8:
            tmp_file.unlink(missing_ok=True)
            for file in f.rglob("*_dynamic_range_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_drq_int8") + file.suffix))
            for file in f.rglob("*_full_integer_quant.tflite"):
                file.rename(file.with_name(file.stem.replace("_full_integer_quant", "_fiq_int8") + file.suffix))
            for file in f.rglob("*_integer_quant_with_int16_act.tflite"):
                file.unlink()  # delete extra fp16 activation TFLite files

        # Add TFLite metadata
        for file in f.rglob("*.tflite"):
            logger.info(f'Model Exporter: {colorstr("TF Lite: ")}{file}.')
            add_tflite_metadata(file, metadata)
        
        logger.info(f'Model Exporter: {prefix} export success, saved as {f}')
    except Exception as e:
        logger.info(f'Model Exporter: {prefix} export failure: {e}')
    return str(f), keras_model


def export_tf_pb(keras_model, file, prefix=colorstr('TensorFlow GraphDef:')):
    # YOLO TensorFlow GraphDef *.pb export https://github.com/leimao/Frozen_Graph_TensorFlow
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    try:
        logger.info(f'\nModel Exporter: {prefix} Starting export with tensorflow {tf.__version__}...')
        f = file.with_suffix('.pb')

        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype))
        frozen_func = convert_variables_to_constants_v2(m)
        frozen_func.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=str(f.parent), name=f.name, as_text=False)
        logger.info(f'Model Exporter: {prefix} export success, saved as {f}')
    except Exception as e:
        logger.info(f'Model Exporter: {prefix} export failure: {e}')
    return f, None


def export_tflite(file, half, prefix=colorstr("TensorFlow Lite:")):
    import tensorflow as tf
    import shutil
    logger.info(f'\nModel Exporter: {prefix} Starting export with tensorflow {tf.__version__}...')
    try:
        saved_model = Path(str(file).replace(file.suffix, "_saved_model"))
        if half:
            f = saved_model / f"{file.stem}_float16.tflite"     # fp16 weights, fp32 in/out
        else:
            f = saved_model / f"{file.stem}_float32.tflite"     # fp32 weights, fp32 in/out
        # f = saved_model / f"{file.stem}_drq_int8.tflite"    # dynamic range quantization: int8 weigths, fp32 in/out
        # f = saved_model / f"{file.stem}_fiq_int8.tflite"    # full integer quantization: int8 weights, int8 in/out
        f_tflite = file.with_suffix('.tflite')
        shutil.copyfile(str(f), f_tflite)
        logger.info(f'Model Exporter: {prefix} export success, {str(f)} is saved as {f_tflite}')
    except Exception as e:
        logger.warning(f'Model Exporter: {prefix} export failure: {e}')
    return f_tflite, None


def export_edgetpu(tflite_model, prefix=colorstr('Edge TPU:')):
    # YOLO Edge TPU export https://coral.ai/docs/edgetpu/models-intro/
    try:
        cmd = 'edgetpu_compiler --version'
        help_url = 'https://coral.ai/docs/edgetpu/compiler/'
        assert platform.system() == 'Linux', f'export only supported on Linux. See {help_url}'
        if subprocess.run(f'{cmd} >/dev/null', shell=True).returncode != 0:
            logger.info(f'\n{prefix} export requires Edge TPU compiler. Attempting install from {help_url}')
            sudo = subprocess.run('sudo --version >/dev/null', shell=True).returncode == 0  # sudo installed on system
            for c in (
                "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -",
                'echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | '
                "sudo tee /etc/apt/sources.list.d/coral-edgetpu.list",
                "sudo apt-get update",
                "sudo apt-get install edgetpu-compiler",
            ):
                subprocess.run(c if sudo else c.replace('sudo ', ''), shell=True, check=True)
        ver = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout.decode().split()[-1]

        logger.info(f'\nModel Exporter: {prefix} starting export with Edge TPU compiler {ver}...')
        f_tpu = str(tflite_model).replace(".tflite", "_edgetpu.tflite")  # Edge TPU model

        cmd = (
            "edgetpu_compiler "
            f'--out_dir {Path(f_tpu).parent} '
            "--show_operations "
            "--search_delegate "
            "--delegate_search_step 30 "
            "--timeout_sec 180 "
            f'{tflite_model}'
        )
        subprocess.run(cmd.split(), check=True)
        logger.info(f'Model Exporter: {prefix} export success, saved as {f_tpu}')
    except Exception as e:
        logger.info(f'Model Exporter: {prefix} export failure: {e}')
    return str(f_tpu), None


def add_tflite_metadata(file, meta):
    # Add metadata to *.tflite models per https://www.tensorflow.org/lite/models/convert/metadata
    # with contextlib.suppress(ImportError):
    import flatbuffers
    try:
        from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
        from tensorflow_lite_support.metadata.python import metadata as _metadata
    except ImportError:
        logger.warning(f'Model Export: {colorstr("TF Lite:")} Import Error')
        # from tflite_support import flatbuffers
        from tflite_support import metadata as _metadata
        from tflite_support import metadata_schema_py_generated as _metadata_fb

    try:
        # tmp_file = Path('/tmp/meta.txt')
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, 'w') as f_meta:
            f_meta.write(str(meta))

        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = meta['description']
        model_meta.version = meta['version']
        model_meta.author = meta['author']
        model_meta.license = meta['license']

        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS

        # model_meta.associatedFiles = [label_file]

        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = 'image'
        input_meta.desciption = 'Input image'
        input_meta.content = _metadata_fb.ContentT()
        input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
        input_meta.content.contentProperties.colorSpace = _metadata_fb.ColorSpaceType.RGB
        input_meta.content.contentPropertiesType = _metadata_fb.ContentProperties.ImageProperties

        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = "output"
        output_meta.description = "Box coordinates and class labels"
        output_meta.associatedFiles = [label_file]

        subgraph = _metadata_fb.SubGraphMetadataT()
        subgraph.inputTensorMetadata = [input_meta]
        subgraph.outputTensorMetadata = [output_meta]
        model_meta.subgraphMetadata = [subgraph]

        b = flatbuffers.Builder(0)
        b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
        metadata_buf = b.Output()

        populator = _metadata.MetadataPopulator.with_model_file(file)
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        tmp_file.unlink()
        logger.warning(f'Model Exporter: {colorstr("TF Lite: ")}successfully add metadata to {file}')
    except Exception as e:
        logger.warning(f'Model Exporter: {colorstr("TF Lite: ")}exception metadata {e}')


def get_int8_calibration_dataloader(data, imgsz, batch, gs, prefix=colorstr("ONNX-to-TF:")):
    """Build and return a dataloader suitable for calibration of INT8 models."""
    from tango.utils.datasets import create_rep_dataloader

    logger.info(f'Model Export: {prefix} collecting INT8 calibration images from data={data["val"]}')
    dataloader, dataset = create_rep_dataloader(data["val"], imgsz[0], batch, gs)
    n = len(dataset)
    logger.info(f'Model Export: {prefix} found {n} images')
    if n < 300:
        logger.warning(f'Model Export: {prefix} more than 300 images recommended for INT8 callibration')
    return dataloader


def export_config(src, dst, data, base, device, engine, task='detection'):
    logger.info(f'\n{colorstr("Model Exporter: ")}Creating meta data...')
    nn_dict = {}

    # NN Model
    config = ''
    if task == 'classification':
        weight = 'bestmodel.pt'
    else: # if task == 'detection'
        weight = [
            'bestmodel.pt', 
            'bestmodel.torchscript', 
            'bestmodel.onnx',
            'bestmodel.tflite',
            'bestmodel_edgetpu.tflite',
        ]
    nn_dict['weight_file'] = weight
    nn_dict['config_file'] = config # not neccessary

    # Label
    nc = data.get('nc')
    ch = data.get('ch', 3) # need to check (ChestXRay: ch=1)
    names = data.get('names')
    if not nc and names:
        nc = len(names)
    if not names and nc:
        names = list(range(nc)) # default name: [0, 1, ..., nc-1]
    nn_dict['nc'] = nc
    nn_dict['names'] = names

    # Input
    imgsz = base.get('imgsz', 640) # need to check (ChestXRay: imgsz=256)
    input_tensor_shape = [1, ch, imgsz, imgsz]
    # device = select_device(device)
    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
    if device.type == 'cpu':
        input_data_type = 'fp32'
    else:
        input_data_type = 'fp16'
    anchors = base.get('anchors', None)
    if (not anchors or anchors == 'None') and task == 'detection':
        # logger.warn(f'Model Exporter: not found anchor imformation')
        logger.info(f'{colorstr("Model Exporter: ")}Anchor-free detection heads')

    nn_dict['input_tensor_shape'] = input_tensor_shape
    nn_dict['input_data_type'] = input_data_type
    nn_dict['anchors'] = anchors

    # Output
    if task == 'detection':
        output_number = 3
        stride = [8, 16, 32]
        need_nms = True
        conf_thres = 0.25
        iou_thres = 0.45
        total_pred_num = 0
        if anchors and (anchors != 'None'): # v7 (num of anchors = 3)
            total_pred_num = sum([3*(imgsz/stride[i])**2 for i in range(output_number)])
            output_size = [
                [1, ch, imgsz/stride[0], imgsz/stride[0], 5+nc],
                [1, ch, imgsz/stride[1], imgsz/stride[1], 5+nc],
                [1, ch, imgsz/stride[2], imgsz/stride[2], 5+nc]
            ]
        else: # v9 (no anchors)
            total_pred_num = sum([(imgsz/stride[i])**2 for i in range(output_number)])
            output_size = [
                [1, 4+nc, int(total_pred_num)], # <= for training
                [1, 4+nc, int(total_pred_num)]  # <= for prediction
            ]
    elif task == 'classification':
        output_number = 1
        output_size = [1, nc]

    nn_dict['output_number'] = output_number
    nn_dict['output_size'] = output_size

    if task == 'detection':
        nn_dict['stride'] = stride
        nn_dict['need_nms'] = need_nms
        nn_dict['conf_thres'] = conf_thres
        nn_dict['iou_thres'] = iou_thres

    # for backward compatibility
    nn_dict['base_dir_autonn'] = 'autonn_core/tango'
    nn_dict['class_file'] = ['common/models/yolo.py',
                             'common/models/common.py',
                             'common/models/experimental.py',
                             'common/models/dynamic_op.py',
                             'common/models/resnet_cifar10.py',
                             'common/models/search_block.py',
                             'common/models/supernet_yolov7.py',
                             'common/models/my_modules.py',
                            ]
    nn_dict['class_name'] = "Model(cfg='basemodel.yaml')"
    # nn_dict['label_info_file'] = None
    # nn_dict['vision_lib'] = None
    # nn_dict['norm'] = None
    # nn_dict['mean'] = None
    # nn_dict['output_format_allow_list'] = None
    if anchors and (anchors != 'None'):
        nn_dict['output_pred_format'] = ['x', 'y', 'w', 'h', 'confidence', 'probability_of_classes']
    else:
        nn_dict['output_pred_format'] = ['x', 'y', 'w', 'h', 'probability_of_classes']

    with open(dst, 'w') as f:
        yaml.dump(nn_dict, f, default_flow_style=False)

    # import pprint
    # print('-'*100)
    # pprint.pprint(nn_dict)
    # print('-'*100)

    logger.info(f'{colorstr("Model Exporter: ")}NN meta information export success, saved as {dst}')
    logger.info('-'*100)


def export_weight(weights, device, include, task='detection', ch=3, imgsz=[640,640]):
    t = time.time()
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, onnx_end2end, xml, engine, saved_model, pb, tflite, edgetpu = flags  # export booleans
    file = Path(weights)

    # options ----------------------------------------------------------------------------------------------------------
    # imgsz = [640, 640]  # input image size
    batch_size = 1      # inference batch size
    inplace = True      # Detect(): set to compute tensors w/o copy
    half = True         # FP16 quantization / GPU only
    int8 = False        # TF: INT8 quantization
    dynamic = False     # ONNX/TF/TensorRT: dynamic input sizes
    simplify = True     # ONNX/ONNX-E2E/TensorRT: simplifies the graph for onnx (TensorRT requires ONNX first)
    opset = 12          # ONNX: operator-set version
    workspace = 4.0     # TensorRT: max space size(GB) for tensorrt
    verbose = False     # TensorRT: detail logging for tensorrt
    nms = False         # TF: add nms
    agnostic_nms = True # TF: add nms to tensorflow
    optimize = False    # TorchScript: mobile optimization / CPU only
    topk_all = 100      # ONNX-E2E: nms -  top-k for all classes to keep
    iou_thres = 0.45    # ONNX-E2E: nms -  iou threshold
    conf_thres = 0.25   # ONNX-E2E: nms -  confidence threshold
    #-------------------------------------------------------------------------------------------------------------------

    # Load PyTorch model
    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')
    if half and device.type == 'cpu':
        logger.warning(f'{colorstr("Model Exporter: ")}--half only compatible with GPU export, ignore --half')
        half = False
    if half and dynamic:
        logger.warning(f'{colorstr("Model Exporter: ")}--half not compatible with --dynamic, ignore --dynamic')
        dynamic = False
    if int8 | edgetpu:
        half = False # onnx2tf: int8 quantization requires fp32 inputs / edgetpu: mappign ops to tpu requires int8 quant 
    model = attempt_load(weights, map_location=device, fused=True)  # load fused FP32 model
    logger.debug(model)

    # Checks
    if edgetpu | tflite:
        imgsz = 320
    imgsz = [imgsz] if isinstance(imgsz, int) else imgsz
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        if device.type != 'cpu':
            logger.warning(f'{colorstr("Model Exporter: ")}--optimize not compatible with cuda devices, ignore --optimize')
            optimize = False

    # Input
    if task == 'detection':
        gs = int(max(model.stride))  # grid size (max stride)
        imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
        im = torch.zeros(batch_size, ch, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection
    elif task == 'classification':
        im = torch.zeros(batch_size, ch, *imgsz).to(device)


    # Update model
    model.eval()
    v9 = False
    for _, m in model.named_modules():
        if isinstance(m, (DDetect, DualDDetect)):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
            if any((saved_model, pb, tflite, edgetpu)):
                m.format = 'tf'
                if int8 | edgetpu:
                    m.format = 'tf-int8'
            v9 = True
        if isinstance(m, (Detect, IDetect, IAuxDetect)):
            # it is for v5/v7
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
    if half:
        im, model = im.half(), model.half()  # to FP16

    if v9:
        '''
        --------------------------------------------------------------------------------------------
        | v9 head       | export | type(y) |             y                                         |
        --------------------------------------------------------------------------------------------
        | DualDDetect   |   T    |  tensor |  (aux-output ⨁ pred-output)*                          |
        |               |   F    |  tuple  | ([aux-output, pred-output*], [aux-train, pred-train]) |
        --------------------------------------------------------------------------------------------
        | DDetect       |   T    |  tensor |     pred-output*                                      |
        |               |   F    |  tuple  |    (pred-output*, pred-train)                         |
        --------------------------------------------------------------------------------------------
        '''
        # shape = tuple((y[0] if isinstance(y, (tuple, list)) else y).shape)  # model output shape
        if isinstance(y, tuple):
            shape = tuple((y[0][-1] if isinstance(y[0], list) else y[0]).shape)
        else:
            shape = tuple((y[-1] if isinstance(y, list) else y).shape)
    else:
        '''
        ------------------------------------------------------------------------------
        | v7 head    | export | end2end | type(y) |              y                   |
        ------------------------------------------------------------------------------
        | IAuxDetect |   T    |    -    |  list   | [p3, p4, p5, p6, a3, a4, a5, a6] |
        |            |   F    |    T    |  tensor |        predict-output            |
        |            |   F    |    F    |  tuple  |    (pred, [p3, p4, p5, p6])      |
        ------------------------------------------------------------------------------
        | IDetect    |   T    |    -    |  list   |          [p3, p4, p5]            |
        |            |   F    |    T    |  tensor |        predict-output            |
        |            |   F    |    F    |  tuple  |      (pred, [p3, p4, p5])        |
        ------------------------------------------------------------------------------
        ------------------------------------------------------------------------------
        | v5 head    | export | end2end | type(y) |              y                   |
        ------------------------------------------------------------------------------   
        | Detect     |   T    |    -    |  list   |          [p3, p4, p5]            |
        |            |   F    |    T    |  tensor |        predict-output            |
        |            |   F    |    F    |  tuple  |      (pred, [p3, p4, p5])        |
        ------------------------------------------------------------------------------
        actually, it must be 'list' or 'tensor'
        '''
        if isinstance(y, tuple):
            shape = tuple(y[0].shape)
        elif isinstance(y, list):
            _shape_list = []
            for yi in y:
                _shape_list.append(yi.shape)
            shape = tuple(_shape_list)
        else:
            shape = tuple(y.shape)
   
    
    if task == 'detection':
        ver = 'v9' if v9 else 'v7'
        metadata = {
            'description': f'TANGO YOLO{ver}-based model',
            'author': 'ETRI',
            'date': datetime.now().isoformat(),
            'version': 'tango-24.11',
            'license': 'GNU General Public License v3.0 or later(GPLv3)',
            'docs': 'https://github.com/ML-TANGO/TANGO/wiki',
            'stride': int(max(model.stride)),
            'task': 'detect',
            'batch': batch_size,
            'imgsz': imgsz,
            'names': model.names
        }  # model metadata
    elif task == 'classification':
        metadata = {
            'description': 'TANGO ResNet-based model',
            'author': 'ETRI',
            'date': datetime.now().isoformat(),
            'version': 'tango-24.11',
            'license': 'GNU General Public License v3.0 or later(GPLv3)',
            'docs': 'https://github.com/ML-TANGO/TANGO/wiki',
            'stride': '',
            'task': 'classify',
            'batch': batch_size,
            'imgsz': imgsz,
            'names': model.names
        }  # model metadata
    logger.info(f"{colorstr('Model Exporter:')} Starting from {file} with output shape {shape} ({os.path.getsize(file) / 1E6:.1f} MB)")

    # Exports
    f = [''] * len(fmts)  # exported filenames
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)  # suppress TracerWarning
    warnings.filterwarnings("ignore", category=FutureWarning) # torch.onnx.__patch_torch.__graph_op will be deprecated
    if jit:  # TorchScript
        f[0], _ = export_torchscript(model, im, file, optimize, task=task)
        logger.info('-'*100)
    if engine:  # TensorRT required ONNX
        f[1], _ = export_tensorrt(model, im, file, half, dynamic, simplify, workspace, verbose=verbose)
        logger.info('-'*100)
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify, task=task)
        logger.info('-'*100)
    if onnx_end2end:
        labels = model.names
        f[3], _ = export_onnx_end2end(model,im,file,simplify,topk_all,iou_thres,conf_thres,device,len(labels),v9)
        logger.info('-'*100)
    if xml:  # OpenVINO
        f[4], _ = export_openvino(file, metadata, half)
        logger.info('-'*100)
    if any((saved_model, pb, tflite, edgetpu)):  # TensorFlow formats
        int8 = int8 | edgetpu # edgetpu mapping reuires the int8 quantized tflite model
        f[5], s_model = export_tf(f[2], metadata, int8, imgsz, batch_size, stride=gs)
        logger.info('-'*100)
        if pb:
            f[6], _ = export_tf_pb(s_model, file)
            logger.info('-'*100)
        if tflite:
            f[7], _ = export_tflite(file, half)
            logger.info('-'*100)
        if edgetpu:
            tflite_model = Path(f[5]) / f"{file.stem}_fiq_int8.tflite"
            if Path.exists(tflite_model):
                f[8], _ = export_edgetpu(str(tflite_model))
                add_tflite_metadata(f[8], metadata)
                import shutil
                shutil.copyfile(f[8], str(file.parent / f"{file.stem}_edgetpu.tflite"))
            else:
                logger.warning(f'Model Exporter: {colorstr("Edge TPU: ")}export failure: not found {str(tflite_model)}')
            logger.info('-'*100)
        # saved_model_dir = Path(str(file).replace('.pt', '_saved_model'))
        # if saved_model_dir.is_dir():
        #     shutil.rmtree(f)  # delete output folder

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        logger.info(f'{colorstr("Model Exporter: ")}Export complete({time.time() - t:.1f}s)')
    return f  # return list of exported files/dirs






def convert_small_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.debug(f"{k}: perfectly matched!!")
    _ = model.eval()
    return model


def convert_medium_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
    _ = model.eval()
    return model


def convert_large_model(model, ckpt):
    idx = 0
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                if "model.{}.".format(idx) in k:
                    break
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                model.state_dict()[k] -= model.state_dict()[k]
                model.state_dict()[k] += ckpt['model'].state_dict()[kr]
                logger.info(f"{k}: perfectly matched!!")
    _ = model.eval()
    return model


def convert_yolov9(model_pt, cfg):
    device = torch.device("cpu")

    if not os.path.isfile(model_pt):
        logger.warning(f'{colorstr("Model Exporter: ")}not found {model_pt}')
        return None

    if not os.path.isfile(cfg):
        logger.warning(f'{colorstr("Model Exporter: ")}not found {cfg}')
        return ckpt
    
    model = Model(cfg, ch=3, nc=80, anchors=3).to(device) # create empty model
    _ = model.eval()

    ckpt = torch.load(model_pt, map_location='cpu')
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    cfg_name = os.path.basename(cfg).lower()
    if '-t' in cfg_name or '-s' in cfg_name:
        model = convert_small_model(model, ckpt)    
    elif '-m' in cfg_name or '-c' in cfg_name:
        model = convert_medium_model(model, ckpt)
    else: #if '-e' in cfg_name:
        model = convert_large_model(model, ckpt)

    reparamed_model = {
        'model' : model,
        'optimizer': ckpt['optimizer'],
        'best_fitness': ckpt['best_fitness'],
        'epoch': ckpt['epoch'],
        'ema': ckpt['ema'],
        'updates': ckpt['updates'],
    }
    # f_path = 'shared / common / uid / pid / autonn / weights / best_converted.pt'
    f_path = str(model_pt).replace('.pt', '_converted.pt')
    logger.info(f'{colorstr("Model Exporter: ")}Converted model is saved as {f_path}')
    torch.save(reparamed_model, f_path)
    return f_path
