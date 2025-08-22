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
                                            DetectionModel,
                                        )
# from tango.utils.dataloaders import LoadImages
from tango.utils.general import (   check_dataset,
                                    check_img_size,
                                    check_requirements,
                                    colorstr,
                                )
from tango.utils.torch_utils import select_device

def _replace_attr(attrs, name, maker):
    for i, a in enumerate(list(attrs)):
        if a.name == name:
            del attrs[i]
            break
    attrs.append(maker)

def patch_averagepool_valid(onnx_path: str, out_path: str = None):
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    changed = False
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            continue
        _replace_attr(node.attribute, "auto_pad", helper.make_attribute("auto_pad", "VALID"))
        _replace_attr(node.attribute, "pads",     helper.make_attribute("pads", [0,0,0,0]))
        changed = True
    if not changed:
        return onnx_path, False
    out = out_path or onnx_path.replace(".onnx", "_avgvalid.onnx")
    onnx.save(model, out)
    return out, True


def patch_resize_attrs(onnx_path: str, out_path: str = None):
    import onnx
    from onnx import helper

    model = onnx.load(onnx_path)
    changed = False
    for node in model.graph.node:
        if node.op_type != "Resize":
            continue
        _replace_attr(node.attribute, "mode",                              helper.make_attribute("mode", "nearest"))
        _replace_attr(node.attribute, "coordinate_transformation_mode",    helper.make_attribute("coordinate_transformation_mode", "asymmetric"))
        _replace_attr(node.attribute, "nearest_mode",                      helper.make_attribute("nearest_mode", "floor"))
        changed = True
    if not changed:
        return onnx_path, False
    out = out_path or onnx_path.replace(".onnx", "_rsattr.onnx")
    onnx.save(model, out)
    return out, True
def _get_valueinfo_chw_map(model):
    def _dimval(d): 
        return d.dim_value if (d.HasField('dim_value') and d.dim_value > 0) else None
    mp = {}
    all_vis = list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)
    for vi in all_vis:
        if not vi.type or not vi.type.tensor_type.shape.dim:
            continue
        dims = vi.type.tensor_type.shape.dim
        C = _dimval(dims[1]) if len(dims) > 1 else None  # NCHW 가정
        H = _dimval(dims[2]) if len(dims) > 2 else None
        W = _dimval(dims[3]) if len(dims) > 3 else None
        mp[vi.name] = (C, H, W)
    return mp

def replace_avgpool_with_dwconv(onnx_path: str) -> str:
    import onnx, numpy as np
    from onnx import helper, TensorProto

    m = onnx.load(onnx_path)
    vi_map = _get_valueinfo_chw_map(m)

    new_nodes = []
    inits = list(m.graph.initializer)
    changed = False

    for node in m.graph.node:
        if node.op_type != "AveragePool":
            new_nodes.append(node)
            continue

        # attrs
        attrs = {a.name: a for a in node.attribute}
        k = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [2,2]
        s = list(attrs["strides"].ints)      if "strides"      in attrs else [2,2]
        pads = list(attrs["pads"].ints)      if "pads"         in attrs else [0,0,0,0]

        x = node.input[0]
        C, H, W = vi_map.get(x, (None, None, None))
        if C is None or C <= 0:
            new_nodes.append(node)
            continue

        kH, kW = int(k[0]), int(k[1])
        w_name = f"{(node.name or 'avg')}_dw_w"
        w_np = (np.ones((C, 1, kH, kW), dtype=np.float32) / float(kH * kW))
        inits.append(helper.make_tensor(w_name, TensorProto.FLOAT, w_np.shape, w_np.flatten().tolist()))

        conv = helper.make_node(
            "Conv",
            inputs=[x, w_name],
            outputs=[node.output[0]],
            name=f"{(node.name or 'avg')}_dwconv",
            strides=s,
            pads=pads,
            group=int(C),
        )
        new_nodes.append(conv)
        changed = True

    if not changed:
        return onnx_path

    m.graph.node[:] = new_nodes
    m.graph.initializer[:] = inits
    out = onnx_path.replace(".onnx", "_avg2dwconv.onnx")
    onnx.save(m, out)
    return out


def _safe_assign(dst_tensor, src_tensor, key_dst, key_src):
    try:
        src = src_tensor
        if src.dtype != dst_tensor.dtype:
            src = src.to(dst_tensor.dtype)
        if src.shape != dst_tensor.shape:
            logger.error(f"[convert] shape mismatch: {key_dst} {tuple(dst_tensor.shape)} != {key_src} {tuple(src.shape)}; skip")
            return False
        dst_tensor.copy_(src)
        return True
    except Exception as e:
        logger.error(f"[convert] assign failed: {key_dst} <- {key_src}: {e}; skip")
        return False


def _ckpt_fetch(sd, key):
    if key in sd:
        return key, sd[key]
    variants = [key]
    if ".cv4.0.0." in key: variants.append(key.replace(".cv4.0.0.", ".cv4."))
    if ".cv3.1."   in key: variants.append(key.replace(".cv3.1.", ".cv3."))
    if ".0.0."     in key: variants.append(key.replace(".0.0.", "."))
    if ".0."       in key: variants.append(key.replace(".0.", "."))
    if ".1."       in key: variants.append(key.replace(".1.", "."))
    seen = set()
    for k2 in variants:
        if k2 not in seen:
            seen.add(k2)
            if k2 in sd:
                return k2, sd[k2]
    return None, None



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
        
        opset = 12

        output_names = ['output0']
        if dynamic:
            dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}
            dynamic['output0'] = {0: 'batch', 1: 'anchors'}

        model_export = model.cpu().float() if dynamic else model.float()
        im_export = im.cpu().float() if dynamic else im.float()

        torch.onnx.export(
            model_export,
            im_export,
            f,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None
        )

        model_onnx = onnx.load(f)
     
        model_onnx.ir_version = 8
        
        onnx.checker.check_model(model_onnx)

        if task == 'detection':
            d = {'stride': int(max(model.stride)), 'names': model.names}
        elif task == 'classification':
            d = {'names': model.names}
        
        for k, v in d.items():
            meta = model_onnx.metadata_props.add()
            meta.key, meta.value = k, str(v)

        if simplify:
            try:
                import onnxsim
                logger.info(f'Model Exporter: {prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                b, c, h, w = [int(x) for x in im.shape]
                inp_name = model_onnx.graph.input[0].name
                overwrite_shapes = {inp_name: [b, c, h, w]}
                
                try:
                    model_onnx, check = onnxsim.simplify(
                        model_onnx,
                        overwrite_input_shapes=overwrite_shapes,
                        skip_shape_inference=True
                    )
                    if check:
                        model_onnx.ir_version = 8
                        logger.info(f'Model Exporter: {prefix} simplifier success')
                except Exception as e:
                    logger.warning(f'Model Exporter: {prefix} simplifier failed: {e}, using original')
                    
            except Exception as e:
                logger.warning(f'Model Exporter: {prefix} simplifier failure: {e}')
        try:
            import onnx
            model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
            logger.info(f'{prefix} shape inference applied')
        except Exception as e:
            logger.warning(f'{prefix} shape inference failed: {e}')
            
        model_onnx.ir_version = 8
        onnx.save(model_onnx, f)
        logger.info(f'Model Exporter: {colorstr("ONNX ")}export success, saved as {f} (IR version: {model_onnx.ir_version})')
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

        model = End2End(
            model, 
            max_obj=topk_all, 
            iou_thres=iou_thres, 
            score_thres=conf_thres, 
            max_wh=None, # None -> ONNX End module for TensorRT, # int -> ONNX End module for ONNX runtime
            device=device, 
            n_classes=labels, 
            v9=v9
        )

        output_names = ['num_dets', 'det_boxes', 'det_scores', 'det_classes']
        shapes = [ batch_size, 1,  batch_size,  topk_all, 4,
                   batch_size,  topk_all,  batch_size,  topk_all]

        torch.onnx.export(  
            model,
            im,
            f,
            verbose=False,
            export_params=True,       # store the trained parameter weights inside the model file
            opset_version=12,
            do_constant_folding=True, # whether to execute constant folding for optimization
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )

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
                logger.warning(f'Model Exporter: {prefix} simplifier failure: {e}')

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
    onnx_path_str, _ = export_onnx(model, im, file, 12, dynamic, simplify)

    try:
        import tensorrt as trt
    except Exception as e:
        logger.warning(f'{prefix} export failure: {e}')
        return None, None

    import pkg_resources as pkg
    current, minimum, maximum = (pkg.parse_version(x) for x in (trt.__version__, '8.0.0', '10.1.0'))
    if minimum >= current or current >= maximum:
        logger.warning(f'{prefix} export failure: {prefix}>={minimum},<={maximum} required by autonn in ml-tango')
        return None, None

    is_trt10 = int(trt.__version__.split(".")[0]) >= 10

    # onnx = file.with_suffix('.onnx')
    from pathlib import Path as _Path
    onnx_path = _Path(onnx_path_str)
    logger.info(f'{prefix} starting export with TensorRT {trt.__version__}...')
    assert onnx_path.exists(), f'failed to export ONNX file: {onnx_path}'
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
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError(f'failed to load ONNX file: {onnx_path}')

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
def _safe_copy(dst_tensor, src_tensor, k, kr):
    src = src_tensor
    if src.dtype != dst_tensor.dtype:
        src = src.to(dst_tensor.dtype)
    if src.shape != dst_tensor.shape:
        logger.error(f"[convert] shape mismatch: {k} {tuple(dst_tensor.shape)} != {kr} {tuple(src.shape)}; skip")
        return
    dst_tensor.copy_(src)

def patch_onnx_resize_static(onnx_path, out_path=None):
    import onnx, onnx_graphsurgeon as gs, numpy as np
    
    model = onnx.load(onnx_path)
    original_ir_version = model.ir_version
    graph = gs.import_onnx(model)
    changed = False
    
    for n in graph.nodes:
        if n.op != "Resize":
            continue
        n.attrs["mode"] = "nearest"
        n.attrs["coordinate_transformation_mode"] = "asymmetric"
        n.attrs["nearest_mode"] = "floor"

        if len(n.inputs) >= 1:
            x = n.inputs[0]
        if len(n.inputs) >= 2:
            n.inputs[1] = gs.Constant(f"{n.name}_roi_empty", np.zeros((0,), np.float32))
        else:
            n.inputs.append(gs.Constant(f"{n.name}_roi_empty", np.zeros((0,), np.float32)))
        if len(n.inputs) >= 3 and not isinstance(n.inputs[2], gs.Constant):
            n.inputs[2] = gs.Constant(f"{n.name}_scales", np.array([1.,1.,2.,2.], np.float32))
            changed = True
        elif len(n.inputs) < 3:
            n.inputs.append(gs.Constant(f"{n.name}_scales", np.array([1.,1.,2.,2.], np.float32)))
            changed = True
        if len(n.inputs) >= 4 and n.inputs[3] is not None:
            n.inputs[3] = None
            changed = True

    graph.cleanup().toposort()
    
    model_fixed = gs.export_onnx(graph)
    model_fixed.ir_version = min(original_ir_version, 8)
    
    out = out_path or onnx_path.replace(".onnx", "_fixed.onnx")
    onnx.save(model_fixed, out)
    return out, changed

def rewrite_resize_to_tile(onnx_path: str, out_path: str = None):
    import onnx, onnx_graphsurgeon as gs, numpy as np

    model = onnx.load(onnx_path)
    graph = gs.import_onnx(model)
    changed = False

    def _c(name, arr):
        return gs.Constant(name=name, values=np.asarray(arr))
    
    def _get_shape(var):
        s = var.shape
        if s is None or any(dim is None for dim in s):
            return None
        return [int(d) for d in s]

    for node in list(graph.nodes):
        if node.op != "Resize":
            continue
        
        if node.attrs.get("mode", "nearest") != "nearest":
            continue

        scales_ok = False
        if len(node.inputs) >= 3 and isinstance(node.inputs[2], gs.Constant):
            s = node.inputs[2].values
            if s is not None and s.size >= 4:
                if float(s[-2]) == 2.0 and float(s[-1]) == 2.0:
                    scales_ok = True
        
        if not scales_ok:
            continue

        x = node.inputs[0]
        shape = _get_shape(x)
        
        if not shape:
            logger.warning(f"Cannot get shape for node {node.name}, skipping")
            continue
        
        ndims = len(shape)
        
        if ndims == 4:
            N, C, H, W = shape
            H2, W2 = H*2, W*2
            
            axes = _c(f"{node.name}_axes", np.array([3, 5], dtype=np.int64))
            unsq_out = gs.Variable(name=f"{node.name}_unsq_out", dtype=None, shape=None)
            unsq = gs.Node("Unsqueeze", name=f"{node.name}_unsq",
                          inputs=[x, axes], outputs=[unsq_out])
            
            repeats = _c(f"{node.name}_repeats", np.array([1, 1, 1, 2, 1, 2], dtype=np.int64))
            tile_out = gs.Variable(name=f"{node.name}_tile_out", dtype=None, shape=None)
            tile = gs.Node("Tile", name=f"{node.name}_tile",
                          inputs=[unsq_out, repeats], outputs=[tile_out])
            
            newshape = _c(f"{node.name}_newshape", np.array([N, C, H2, W2], dtype=np.int64))
            reshape = gs.Node("Reshape", name=f"{node.name}_reshape",
                             inputs=[tile_out, newshape], outputs=node.outputs)
            
            graph.nodes += [unsq, tile, reshape]
            node.outputs = []
            changed = True
            
        elif ndims == 6:
            repeats_values = np.ones(ndims, dtype=np.int64)
            repeats_values[-2] = 2
            repeats_values[-1] = 2
            
            repeats = _c(f"{node.name}_repeats", repeats_values)
            tile_out = gs.Variable(name=f"{node.name}_tile_out", dtype=None, shape=None)
            tile = gs.Node("Tile", name=f"{node.name}_tile",
                          inputs=[x, repeats], outputs=[tile_out])
            
            out_shape = list(shape)
            out_shape[-2] *= 2
            out_shape[-1] *= 2
            
            newshape = _c(f"{node.name}_newshape", np.array(out_shape, dtype=np.int64))
            reshape = gs.Node("Reshape", name=f"{node.name}_reshape",
                             inputs=[tile_out, newshape], outputs=node.outputs)
            
            graph.nodes += [tile, reshape]
            node.outputs = []
            changed = True
        else:
            logger.warning(f"Unsupported tensor dimensions {ndims} for node {node.name}")
            continue

    if changed:
        graph.cleanup().toposort()
        graph.opset = max(13, getattr(graph, "opset", 13))
        out = out_path or onnx_path.replace(".onnx", "_fixed_tile.onnx")
        onnx.save(gs.export_onnx(graph), out)
        logger.info(f"Tile rewrite successful: {out}")
        return out, True
    
    return onnx_path, False

def simplify_onnx_file(path: str) -> str:
    import onnx
    import onnxsim
    
    try:
        m = onnx.load(path)
        
        if m.ir_version > 9:
            logger.warning(f"ONNX IR version {m.ir_version} is higher than supported, attempting to continue...")
        
        try:
            m_s, ok = onnxsim.simplify(m)
            assert ok, "onnxsim.simplify failed"
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}, using original model")
            return path
        
        out = path.replace(".onnx", "_sim.onnx")
        onnx.save(m_s, out)
        return out
    except Exception as e:
        logger.warning(f"ONNX file simplification error: {e}")
        return path

def export_tf_direct(f_onnx, metadata, int8, imgsz, batch_size, stride=1, prefix=colorstr("TF Saved Model:")):
    import numpy as np
    import onnx2tf
    import onnx
    import shutil
    import subprocess
    from pathlib import Path

    logger.info(f"\nModel Exporter: {prefix} Direct conversion with pre-patches...")

    f_onnx = str(f_onnx)
    f = Path(f_onnx.replace('.onnx', '_saved_model'))
    if f.is_dir():
        shutil.rmtree(f)

    try:
        model = onnx.load(f_onnx)
        if model.ir_version > 8:
            logger.info(f"Adjusting IR version from {model.ir_version} to 8")
            model.ir_version = 8
            temp_onnx = f_onnx.replace('.onnx', '_ir8.onnx')
            onnx.save(model, temp_onnx)
            f_onnx = temp_onnx
    except Exception as e:
        logger.warning(f"Could not adjust IR version: {e}")

    try:
        import onnx
        _m = onnx.load(str(f_onnx))
        _m = onnx.shape_inference.infer_shapes(_m)
        tmp_shapes = str(f_onnx).replace('.onnx', '_shapes.onnx')
        onnx.save(_m, tmp_shapes)
        f_onnx = tmp_shapes
        logger.info(f"{prefix} applied ONNX shape inference pre-patch: {f_onnx}")
    except Exception as e:
        logger.warning(f"{prefix} pre-patch shape inference failed: {e}")

    try:
        f_onnx, ch1 = patch_averagepool_valid(f_onnx)
        if ch1: logger.info(f"{prefix} AveragePool auto_pad→VALID patched: {f_onnx}")
    except Exception as e:
        logger.warning(f"{prefix} AveragePool patch skipped: {e}")
    try:
        f_onnx, ch2 = patch_resize_attrs(f_onnx)
        if ch2: logger.info(f"{prefix} Resize attrs patched: {f_onnx}")
    except Exception as e:
        logger.warning(f"{prefix} Resize patch skipped: {e}")

        
    patched_used = False
    try:
        try:
            patched_path, changed = patch_onnx_resize_static(f_onnx)
            if changed:
                logger.info(f"{prefix} Applied Resize(static) patch: {patched_path}")
                f_onnx = patched_path
                patched_used = True
        except Exception as e:
            logger.warning(f"{prefix} patch_onnx_resize_static skipped: {e}")

        if not patched_used:
            try:
                patched_path, changed = rewrite_resize_to_tile(f_onnx)
                if changed:
                    logger.info(f"{prefix} Rewrote Resize→Tile: {patched_path}")
                    f_onnx = patched_path
                    patched_used = True
            except Exception as e:
                logger.warning(f"{prefix} rewrite_resize_to_tile skipped: {e}")
    except Exception as e:
        logger.warning(f"{prefix} pre-patch stage warning: {e}")

    if isinstance(imgsz, (list, tuple)):
        H, W = int(imgsz[0]), int(imgsz[1])
    else:
        H = W = int(imgsz)

    cmd = [
        "onnx2tf",
        "-i", str(f_onnx),
        "-o", str(f),
        "-b", str(batch_size),
        "-ois", f"images:{batch_size},3,{H},{W}",
        "--not_use_onnxsim",
        "-osd",
        "--non_verbose"
    ]
    if int8:
        cmd.extend(["-oiqt", "-qt", "per-tensor"])

    logger.info(f"{prefix} Command: {' '.join(cmd)}")

    def _tail(s: str, n: int = 2000) -> str:
        return (s or "")[-n:]

    try:
        env = os.environ.copy()
        env['TF_ENABLE_ONEDNN_OPTS'] = '0'
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)

        if result.returncode != 0:
            stdout_tail = _tail(result.stdout)
            stderr_tail = _tail(result.stderr)
            logger.error(f"{prefix} Conversion failed (code={result.returncode}).")
            logger.error(f"{prefix} stdout tail:\n{stdout_tail}")
            logger.error(f"{prefix} stderr tail:\n{stderr_tail}")

            logger.info("Retrying with minimal options...")
            cmd_simple = ["onnx2tf", "-i", str(f_onnx), "-o", str(f), "--not_use_onnxsim"]
            result2 = subprocess.run(cmd_simple, capture_output=True, text=True, env=env, timeout=300)

            if result2.returncode != 0:
                stdout_tail2 = _tail(result2.stdout)
                stderr_tail2 = _tail(result2.stderr)
                logger.error(f"{prefix} Minimal conversion failed (code={result2.returncode}).")
                logger.error(f"{prefix} stdout tail:\n{stdout_tail2}")
                logger.error(f"{prefix} stderr tail:\n{stderr_tail2}")
                raise Exception("Direct onnx2tf conversion failed in both attempts.")

        logger.info(f"{prefix} Conversion successful")

    except subprocess.TimeoutExpired:
        logger.error(f"{prefix} Conversion timeout after 300 seconds")
        raise
    except Exception as e:
        raise

    try:
        with open(f / "metadata.yaml", 'w') as yaml_file:
            yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()},
                           yaml_file, sort_keys=False)
        logger.info(f'Model Exporter: Metadata saved')
    except Exception as e:
        logger.warning(f"Metadata save failed: {e}")

    if int8:
        for file in f.rglob("*_dynamic_range_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_dynamic_range_quant", "_drq_int8") + file.suffix))
        for file in f.rglob("*_full_integer_quant.tflite"):
            file.rename(file.with_name(file.stem.replace("_full_integer_quant", "_fiq_int8") + file.suffix))

    for file in f.rglob("*.tflite"):
        logger.info(f'Model Exporter: {colorstr("TF Lite: ")}{file}')
        add_tflite_metadata(file, metadata)

    logger.info(f'Model Exporter: {prefix} Export complete, saved as {f}')
    return str(f), None


def export_tf_fallback(f_onnx, metadata, int8, imgsz, batch_size, stride=1, prefix=colorstr("TF Saved Model (Fallback):")):
    import onnx2tf
    import shutil
    
    logger.info(f"{prefix} Using original ONNX without Tile conversion...")
    
    original_onnx = str(f_onnx).replace("_fixed_fixed_tile.onnx", ".onnx")
    if not Path(original_onnx).exists():
        original_onnx = f_onnx
    
    f = Path(str(original_onnx).replace('.onnx', '_saved_model'))
    if f.is_dir():
        shutil.rmtree(f)
    
    try:
        keras_model = onnx2tf.convert(
            input_onnx_file_path=original_onnx,
            output_folder_path=str(f),
            not_use_onnxsim=True,
            verbosity="info",
            disable_group_convolution=True,
        )
        
        logger.info(f"{prefix} Fallback conversion successful")
        return str(f), keras_model
    except Exception as e:
        logger.error(f"{prefix} Fallback also failed: {e}")
        raise

def export_tf_simple(f_onnx, metadata, int8, imgsz, batch_size, stride=1, prefix=colorstr("TF Saved Model (Simple):")):
    import onnx2tf
    import shutil
    
    logger.info(f"{prefix} Using minimal conversion options...")
    
    original_onnx = str(f_onnx).replace("_fixed_fixed_tile.onnx", ".onnx")
    if not Path(original_onnx).exists():
        original_onnx = f_onnx
    
    f = Path(str(original_onnx).replace('.onnx', '_saved_model'))
    if f.is_dir():
        shutil.rmtree(f)
    
    try:
        cmd = [
            "onnx2tf",
            "-i", str(original_onnx),
            "-o", str(f),
            "--not_use_onnxsim"
        ]
        
        logger.info(f"{prefix} Simple CLI: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"{prefix} Simple conversion successful")
            return str(f), None
        else:
            logger.error(f"{prefix} Simple conversion failed: {result.stderr}")
            raise Exception("All conversion attempts failed")
            
    except Exception as e:
        logger.error(f"{prefix} Simple conversion failed: {e}")
        raise

def export_tf_original(f_onnx, metadata, int8, imgsz, batch_size, stride=1, 
                       prefix=colorstr("TF Saved Model (Original):")):

    import shutil
    
    logger.info(f"{prefix} Using original ONNX without any patches...")
    
    original_onnx = str(f_onnx).replace("_fixed.onnx", ".onnx")
    if not Path(original_onnx).exists():
        original_onnx = f_onnx
    
    f = Path(str(original_onnx).replace('.onnx', '_saved_model'))
    if f.is_dir():
        shutil.rmtree(f)
    
    cmd = [
        "onnx2tf",
        "-i", str(original_onnx),
        "-o", str(f),
        "-b", str(batch_size),
        "-ois", f"images:{batch_size},3,{imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz},{imgsz[1] if isinstance(imgsz, (list, tuple)) else imgsz}",
        "--not_use_onnxsim",
        "-osd"
    ]
    
    logger.info(f"{prefix} Simple CLI: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"{prefix} Original ONNX conversion successful")
            return str(f), None
        else:
            logger.error(f"{prefix} Conversion failed: {result.stderr[-1000:]}")
            if f.exists():
                logger.warning(f"{prefix} Partial conversion may have succeeded, continuing...")
                return str(f), None
            raise Exception("All conversion attempts failed")
            
    except subprocess.TimeoutExpired:
        logger.error(f"{prefix} Conversion timeout")
        raise Exception("Conversion timeout")
    except Exception as e:
        logger.error(f"{prefix} Final conversion failed: {e}")
        raise


def export_tf(f_onnx, metadata, int8, imgsz, batch_size, stride=1, prefix=colorstr("TF Saved Model:")):
    import os
    import numpy as np
    import onnx2tf
    import onnx
    import yaml
    import shutil
    import subprocess
    from pathlib import Path

    keras_model = None
    logger.info(f"\nModel Exporter: {prefix} Starting export with onnx2tf {onnx2tf.__version__}...")

    out_dir = Path(str(f_onnx).replace('.onnx', '_saved_model'))
    if out_dir.is_dir():
        shutil.rmtree(out_dir)

    orig_onnx = str(f_onnx)
    used_onnx = orig_onnx

    def _log_cmd(cmd_list):
        import shlex
        return ' '.join(shlex.quote(x) for x in cmd_list)

    def _onnx_save(model, path):
        onnx.save(model, path)
        return path

    def _shape_infer(path):
        try:
            m = onnx.load(path)
            m = onnx.shape_inference.infer_shapes(m)
            out = path.replace('.onnx', '_shapes.onnx')
            onnx.save(m, out)
            logger.info(f"{prefix} applied ONNX shape inference: {out}")
            return out, m
        except Exception as e:
            logger.warning(f"{prefix} shape inference failed: {e}")
            return path, None

    def _get_hw_c_from_value_info(model):
        def _dimval(d):
            return d.dim_value if (d.HasField('dim_value') and d.dim_value > 0) else None
        mp = {}
        for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
            if not vi.type or not vi.type.tensor_type.shape.dim:
                continue
            dims = vi.type.tensor_type.shape.dim
            C = _dimval(dims[1]) if len(dims) > 1 else None
            H = _dimval(dims[2]) if len(dims) > 2 else None
            W = _dimval(dims[3]) if len(dims) > 3 else None
            mp[vi.name] = (C, H, W)
        return mp

    def _supports_keep_transpose(ver: str) -> bool:
        try:
            parts = [int(x) for x in ver.split(".")]
            return (parts[0] > 1) or (parts[0] == 1 and parts[1] >= 17)
        except Exception:
            return True

    try:
        m0 = onnx.load(used_onnx)
        if m0.ir_version > 8:
            logger.warning(f"ONNX IR version {m0.ir_version} is too high, downgrading to 8")
            m0.ir_version = 8
            used_onnx = _onnx_save(m0, used_onnx)
    except Exception as e:
        logger.warning(f"Could not adjust IR version: {e}")

    try:
        m_tmp = onnx.load(used_onnx)
        inp_name = m_tmp.graph.input[0].name
    except Exception:
        inp_name = "images"
    if isinstance(imgsz, (list, tuple)):
        H, W = int(imgsz[0]), int(imgsz[1])
    else:
        H = W = int(imgsz)
    ois = f"{inp_name}:{batch_size},3,{H},{W}"

    if int8:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = out_dir / "tmp_tflite_int8_calibration_images.npy"
        data_yaml = str(DATASET_ROOT / 'coco128/dataset.yaml')
        if os.path.isfile(data_yaml):
            with open(data_yaml) as representative_data:
                data = yaml.load(representative_data, Loader=yaml.SafeLoader)
            images = []
            dataloader = get_int8_calibration_dataloader(data, imgsz, batch_size, stride)
            for im, _, _, _ in dataloader:
                images.append(im)
            images = torch.nn.functional.interpolate(torch.cat(images, 0).float(), size=imgsz).permute(0, 2, 3, 1)
            np.save(str(tmp_file), images.numpy().astype(np.float32))

    used_onnx, model_inferred = _shape_infer(used_onnx)
    if model_inferred is None:
        try:
            model_inferred = onnx.load(used_onnx)
        except:
            model_inferred = None

    def patch_avgpool_strict(path_in: str) -> str:
        from onnx import helper
        m = onnx.load(path_in)
        changed = False
        for node in m.graph.node:
            if node.op_type != "AveragePool":
                continue
            node.attribute[:] = [a for a in node.attribute if a.name != "auto_pad"]

            attrs = {a.name: a for a in node.attribute}
            if "kernel_shape" not in attrs:
                node.attribute.append(helper.make_attribute("kernel_shape", [2, 2]))
            if "strides" not in attrs:
                node.attribute.append(helper.make_attribute("strides", [2, 2]))

            node.attribute[:] = [a for a in node.attribute if a.name != "pads"]
            node.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))
            node.attribute[:] = [a for a in node.attribute if a.name != "ceil_mode"]
            node.attribute.append(helper.make_attribute("ceil_mode", 0))
            changed = True

        if not changed:
            return path_in
        out = path_in.replace(".onnx", "_avgfix.onnx")
        onnx.save(m, out)
        logger.info(f"{prefix} AveragePool strict-patched: {out}")
        return out

    def patch_resize_attrs(path_in: str) -> str:
        from onnx import helper
        m = onnx.load(path_in)
        changed = False
        for node in m.graph.node:
            if node.op_type != "Resize":
                continue
            def _repl(name, val):
                node.attribute[:] = [a for a in node.attribute if a.name != name]
                node.attribute.append(helper.make_attribute(name, val))
            _repl("mode", "nearest")
            _repl("coordinate_transformation_mode", "asymmetric")
            _repl("nearest_mode", "floor")
            changed = True
        if not changed:
            return path_in
        out = path_in.replace(".onnx", "_rsattr.onnx")
        onnx.save(m, out)
        logger.info(f"{prefix} Resize attrs patched: {out}")
        return out

    used_onnx = patch_avgpool_strict(used_onnx)
    used_onnx = patch_resize_attrs(used_onnx)

    def replace_avgpool_with_dwconv(path_in: str) -> str:
        import numpy as np
        from onnx import helper, TensorProto
        m = onnx.load(path_in)
        shape_map = _get_hw_c_from_value_info(m)
        new_nodes = []
        init_list = list(m.graph.initializer)
        changed = False

        for node in m.graph.node:
            if node.op_type != "AveragePool":
                new_nodes.append(node)
                continue

            attrs = {a.name: a for a in node.attribute}
            k = list(attrs["kernel_shape"].ints) if "kernel_shape" in attrs else [2, 2]
            s = list(attrs["strides"].ints)      if "strides"      in attrs else [2, 2]
            pads = list(attrs["pads"].ints)      if "pads"         in attrs else [0, 0, 0, 0]

            x_name = node.input[0]
            C, _, _ = shape_map.get(x_name, (None, None, None))
            if C is None or C <= 0:
                new_nodes.append(node)
                continue

            kH, kW = int(k[0]), int(k[1])
            w_name = f"{node.name or 'avg'}_dw_w"
            w_arr = (np.ones((C, 1, kH, kW), dtype=np.float32) / float(kH * kW))
            init_list.append(helper.make_tensor(w_name, TensorProto.FLOAT, w_arr.shape, w_arr.flatten().tolist()))

            conv = helper.make_node(
                "Conv",
                inputs=[x_name, w_name],
                outputs=[node.output[0]],
                name=f"{node.name or 'avg'}_dwconv",
                strides=s,
                pads=pads,
                group=int(C)
            )
            new_nodes.append(conv)
            changed = True

        if not changed:
            return path_in

        m.graph.node[:] = new_nodes
        m.graph.initializer[:] = init_list
        out = path_in.replace(".onnx", "_avg2dwconv.onnx")
        onnx.save(m, out)
        logger.info(f"{prefix} AveragePool→DepthwiseConv rewritten: {out}")
        return out
    def run_onnx2tf(onnx_path: str, out_folder: Path, use_keep_transpose: bool) -> bool:
        cmd = [
            "onnx2tf", "-i", str(onnx_path), "-o", str(out_folder),
            "--non_verbose", "--not_use_onnxsim",
            "--disable_group_convolution", "--enable_batchmatmul_unfold",
            "-b", str(batch_size),
            "-ois", ois,
            "-osd",
        ]
        if use_keep_transpose:
            cmd.append("-kt")
        if int8:
            cmd += ["-oiqt", "-qt", "per-tensor"]

        logger.info(f"{prefix} CLI: {_log_cmd(cmd)}")
        env = os.environ.copy()
        env['TF_ENABLE_ONEDNN_OPTS'] = '0'
        env['TF_CPP_MIN_LOG_LEVEL'] = '2'

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        for line in proc.stdout:
            line = line.rstrip()
            if "ERROR" in line or "error" in line.lower():
                logger.error(f"[onnx2tf] {line}")
            else:
                logger.debug(f"[onnx2tf] {line}")
        ret = proc.wait()
        return ret == 0

    kt_ok = _supports_keep_transpose(onnx2tf.__version__)

    ok = run_onnx2tf(used_onnx, out_dir, use_keep_transpose=kt_ok)
    if not ok:
        logger.error(f"onnx2tf failed on strict-patched ONNX (with -kt={kt_ok}) → trying without -kt")
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        ok2 = run_onnx2tf(used_onnx, out_dir, use_keep_transpose=False)

        if not ok2:
            logger.error(f"onnx2tf failed again → rewriting AveragePool to DepthwiseConv")
            used_onnx = replace_avgpool_with_dwconv(used_onnx)
            if out_dir.is_dir():
                shutil.rmtree(out_dir)

            ok3 = run_onnx2tf(used_onnx, out_dir, use_keep_transpose=kt_ok)
            if not ok3:
                logger.error(f"onnx2tf failed on DWConv-rewritten ONNX (with -kt={kt_ok}) → trying without -kt")
                if out_dir.is_dir():
                    shutil.rmtree(out_dir)
                ok4 = run_onnx2tf(used_onnx, out_dir, use_keep_transpose=False)

                if not ok4:
                    logger.info(f"{prefix} Trying alternative: using original ONNX without patches...")
                    return export_tf_original(orig_onnx, metadata, int8, imgsz, batch_size, stride, prefix)

    try:
        with open(out_dir / "metadata.yaml", 'w') as yaml_file:
            yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()},
                           yaml_file, sort_keys=False)
        logger.info(f'Model Exporter: {colorstr("TF Lite: ")}external meta file saved.')
    except Exception as _e:
        logger.warning(f"{prefix} metadata.yaml write failed: {_e}")

    for tfile in out_dir.rglob("*.tflite"):
        logger.info(f'Model Exporter: {colorstr("TF Lite: ")}{tfile}.')
        add_tflite_metadata(tfile, metadata)

    logger.info(f'Model Exporter: {prefix} export success, saved as {out_dir}')
    return str(out_dir), keras_model


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
            f = saved_model / f"{file.stem}_float16.tflite"
        else:
            f = saved_model / f"{file.stem}_float32.tflite"
            
        if not f.exists():
            alternatives = [
                saved_model / f"{file.stem}_float32.tflite",
                saved_model / f"{file.stem}_float16.tflite",
                saved_model / f"{file.stem}_drq_int8.tflite",
                saved_model / f"{file.stem}_fiq_int8.tflite",
            ]
            for alt in alternatives:
                if alt.exists():
                    f = alt
                    logger.info(f'{prefix} Using alternative: {alt.name}')
                    break
                else:
                    tflite_files = list(saved_model.glob("*.tflite"))
                    if tflite_files:
                        f = tflite_files[0]
                        logger.info(f'{prefix} Using first available TFLite: {f.name}')
                    else:
                        import tensorflow as tf
                        logger.info(f'{prefix} No TFLite in {saved_model}, converting from SavedModel...')
                        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
                        if half:
                            converter.optimizations = [tf.lite.Optimize.DEFAULT]
                            converter.target_spec.supported_types = [tf.float16]
                        tflite_bytes = converter.convert()
                        out_local = saved_model / f"{file.stem}_{'float16' if half else 'float32'}.tflite"
                        with open(out_local, "wb") as _f:
                            _f.write(tflite_bytes)
                        f = out_local
        
        f_tflite = file.with_suffix('.tflite')
        shutil.copyfile(str(f), f_tflite)
        logger.info(f'Model Exporter: {prefix} export success, {str(f)} is saved as {f_tflite}')
        return f_tflite, None
        
    except Exception as e:
        logger.warning(f'Model Exporter: {prefix} export failure: {e}')
        return None, None
def export_pb_from_saved_model(saved_model_dir: Path, file, prefix=colorstr('TensorFlow GraphDef:')):
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    try:
        logger.info(f'\nModel Exporter: {prefix} Freezing from SavedModel...')
        f_pb = file.with_suffix('.pb')
        loaded = tf.saved_model.load(str(saved_model_dir))
        if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
            concrete = loaded.signatures['serving_default']
        else:
            sigs = list(getattr(loaded, 'signatures', {}).values())
            if not sigs:
                raise RuntimeError("No signatures found in SavedModel")
            concrete = sigs[0]
        frozen = convert_variables_to_constants_v2(concrete)
        tf.io.write_graph(frozen.graph, str(f_pb.parent), f_pb.name, as_text=False)
        logger.info(f'Model Exporter: {prefix} export success, saved as {f_pb}')
        return f_pb, None
    except Exception as e:
        logger.info(f'Model Exporter: {prefix} export failure: {e}')
        return None, None

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
    try:
        try:
            from tensorflow_lite_support.metadata import metadata_schema_py_generated as _metadata_fb
            from tensorflow_lite_support.metadata.python import metadata as _metadata
            use_tf_support = True
        except ImportError:
            try:
                from tflite_support import metadata as _metadata
                from tflite_support import metadata_schema_py_generated as _metadata_fb
                use_tf_support = False
            except ImportError:
                logger.warning(f'Model Export: {colorstr("TF Lite:")} Metadata support libraries not found. Skipping metadata addition.')
                logger.warning(f'To enable metadata, install: pip install tflite-support or tensorflow-lite-support')
                return
        
        import flatbuffers
        
        tmp_file = Path(file).parent / "temp_meta.txt"
        with open(tmp_file, 'w') as f_meta:
            f_meta.write(str(meta))

        model_meta = _metadata_fb.ModelMetadataT()
        model_meta.name = meta.get('description', 'TANGO Model')
        model_meta.version = meta.get('version', '1.0.0')
        model_meta.author = meta.get('author', 'ETRI')
        model_meta.license = meta.get('license', 'GPL-3.0')

        label_file = _metadata_fb.AssociatedFileT()
        label_file.name = tmp_file.name
        label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
        
        input_meta = _metadata_fb.TensorMetadataT()
        input_meta.name = 'image'
        input_meta.description = 'Input image'
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

        populator = _metadata.MetadataPopulator.with_model_file(str(file))
        populator.load_metadata_buffer(metadata_buf)
        populator.load_associated_files([str(tmp_file)])
        populator.populate()
        
        tmp_file.unlink(missing_ok=True)
        
        logger.info(f'Model Exporter: {colorstr("TF Lite: ")}Successfully added metadata to {file}')
        
    except Exception as e:
        logger.warning(f'Model Exporter: {colorstr("TF Lite: ")}Failed to add metadata: {e}')
        logger.warning(f'Model will work without metadata. To fix, install: pip install tflite-support')

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
        weight = 'bestmodel.pt' # 'bestmodel.torchscript', 'bestmodel.onnx'
    else: # if task == 'detection'
        weight = [
            'bestmodel.pt', 
            'bestmodel.torchscript', 
            'bestmodel.onnx'
        ]
        if engine == 'tensorrt':
            weight.append('bestmodel_end2end.onnx')
        elif engine == 'tflite':
            tfmodels = ['bestmodel.pb', 'bestmodel.tflite']
            weight.extend(tfmodels)
            if device == 'tpu':
                weight.append('bestmodel_edgetpu.tflite')
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
    if task == 'detection':
        nn_dict['class_name'] = "Model(cfg='basemodel.yaml')"
    elif task == 'classification':
        nn_dict['class_name'] = "ClassifyModel(cfg='basemodel.yaml')"
    # nn_dict['label_info_file'] = None
    # nn_dict['vision_lib'] = None
    # nn_dict['norm'] = None
    # nn_dict['mean'] = None
    # nn_dict['output_format_allow_list'] = None

    if task == 'detection':
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
    s_model = None
    onnx_path = None
    t = time.time()
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    jit, onnx, onnx_end2end, xml, engine, saved_model, pb, tflite, edgetpu = flags
    file = Path(weights)

    batch_size = 1
    inplace = True
    half = False
    int8 = False
    dynamic = False
    simplify = True
    opset = 12
    workspace = 4.0
    verbose = False
    nms = False
    agnostic_nms = True
    optimize = False
    topk_all = 100
    iou_thres = 0.45
    conf_thres = 0.25

    device = torch.device('cuda:0' if device == 'cuda' else 'cpu')    
    if device.type == 'cpu':
        half = False
        logger.info(f'{colorstr("Model Exporter: ")}Using FP32 for CPU export')
    
    model = attempt_load(weights, map_location=device, fused=True)
    model = model.float().eval()
    
    if edgetpu | tflite:
        imgsz = 320
    imgsz = [imgsz] if isinstance(imgsz, int) else imgsz
    imgsz *= 2 if len(imgsz) == 1 else 1
    
    if task == 'detection':
        gs = int(max(model.stride))
        imgsz = [check_img_size(x, gs) for x in imgsz]
        im = torch.zeros(batch_size, ch, *imgsz).to(device).float()
    elif task == 'classification':
        im = torch.zeros(batch_size, ch, *imgsz).to(device).float()
        
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
            else:
                m.format = 'pytorch'
            v9 = True
        if isinstance(m, (Detect, IDetect, IAuxDetect)):
            
            m.inplace = inplace
            m.dynamic = dynamic
            if onnx_end2end:
                m.end2end = True
                m.export = False
            else:
                m.end2end = False
                m.export = True

    for _ in range(2):
        y = model(im)
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
    if onnx or xml:
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify, task=task)
        onnx_path = f[2]
    if onnx_end2end:
        labels = model.names
        f[3], _ = export_onnx_end2end(model,im,file,simplify,topk_all,iou_thres,conf_thres,device,len(labels),v9)
        logger.info('-'*100)
    if xml:
        f[4], _ = export_openvino(file, metadata, half)
        logger.info('-'*100)
    if any((saved_model, pb, tflite, edgetpu)) and not onnx_path:
        onnx_path, _ = export_onnx(model, im, file, opset, dynamic, simplify, task=task)
    if any((saved_model, pb, tflite, edgetpu)):
        int8 = int8 | edgetpu
        onnx_path = f[2]
        use_onnx = onnx_path
        
        try:
            f[5], s_model = export_tf(use_onnx, metadata, int8, imgsz, batch_size, stride=gs)
        except Exception as e:
            logger.error(f"{colorstr('TF Saved Model:')} robust converter failed. Reason: {e}")
            try:
                f[5], s_model = export_tf_direct(use_onnx, metadata, int8, imgsz, batch_size, stride=gs)
            except Exception as e2:
                logger.error(f"{colorstr('TF Saved Model:')} direct path also failed. Reason: {e2}")
                f[5], s_model = export_tf_original(use_onnx, metadata, int8, imgsz, batch_size, stride=gs)

        # PB
        if pb:
            saved_model_dir = Path(f[5])
            if s_model is not None:
                f[6], _ = export_tf_pb(s_model, file)
            else:
                f[6], _ = export_pb_from_saved_model(saved_model_dir, file)
            logger.info('-'*100)

        # TFLite
        if tflite:
            f[7], _ = export_tflite(file, half)
            logger.info('-'*100)

        # EdgeTPU
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
    idx, cnt = 0, 0
    max = len(model.state_dict().items())
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                ten_percent_cnt = int((cnt+1)/max*10+0.5)
                bar = '|'+ '🟩'*ten_percent_cnt + ' '*(20-ten_percent_cnt*2)+'|'
                s = f'model.{idx:2.0f} weights transferring...'
                s += (f'{bar} {(cnt+1)/max*100:3.0f}% {cnt+1:6.0f}/{max:6.0f}')
                logger.info(s)
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        cnt += 1
    _ = model.eval()
    return model


def convert_medium_model(model, ckpt):
    idx, cnt = 0, 0
    max = len(model.state_dict().items())
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                ten_percent_cnt = int((cnt+1)/max*10+0.5)
                bar = '|'+ '🟩'*ten_percent_cnt + ' '*(20-ten_percent_cnt*2)+'|'
                s = f'model.{idx:2.0f} weights transferring...'
                s += (f'{bar} {(cnt+1)/max*100:3.0f}% {cnt+1:6.0f}/{max:6.0f}')
                logger.debug(s)
                if "model.{}.".format(idx) in k:
                    break
            if idx < 22:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+1))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+16))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        cnt += 1
    _ = model.eval()
    return model


def convert_large_model(model, ckpt):
    idx, cnt = 0, 0
    max = len(model.state_dict().items())
    for k, v in model.state_dict().items():
        if "model.{}.".format(idx) in k:
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        else:
            while True:
                idx += 1
                ten_percent_cnt = int((cnt+1)/max*10+0.5)
                bar = '|'+ '🟩'*ten_percent_cnt + ' '*(20-ten_percent_cnt*2)+'|'
                s = f'model.{idx:2.0f} weights transferring...'
                s += (f'{bar} {(cnt+1)/max*100:3.0f}% {cnt+1:6.0f}/{max:6.0f}')
                logger.info(s)
                if "model.{}.".format(idx) in k:
                    break
            if idx < 29:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif idx < 42:
                kr = k.replace("model.{}.".format(idx), "model.{}.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv2.".format(idx) in k:
                kr = k.replace("model.{}.cv2.".format(idx), "model.{}.cv4.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.cv3.".format(idx) in k:
                kr = k.replace("model.{}.cv3.".format(idx), "model.{}.cv5.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
            elif "model.{}.dfl.".format(idx) in k:
                kr = k.replace("model.{}.dfl.".format(idx), "model.{}.dfl2.".format(idx+7))
                dst = model.state_dict()[k]
                ckpt_sd = ckpt['model'].state_dict() if ('model' in ckpt and hasattr(ckpt['model'], 'state_dict')) else ckpt.get('state_dict', ckpt)
                kr_found, src = _ckpt_fetch(ckpt_sd, kr)
                if src is None:
                    logger.error(f"[convert] missing key in ckpt: {kr}; skip")
                else:
                    _safe_assign(dst, src, k, kr_found)
                logger.debug(f"{k}: perfectly matched!!")
        cnt += 1
    _ = model.eval()
    return model


def convert_yolov9(model_pt, cfg):
    import traceback
    device = torch.device("cpu")

    if not os.path.isfile(model_pt):
        logger.warning(f'{colorstr("Model Exporter: ")}not found {model_pt}')
        return None

    model_sizes = ['t', 's', 'm', 'c', 'e', 'supernet']
    detected_size = 't'
    
    cfg_lower = cfg.lower()
    for size in model_sizes:
        if f'yolov9-{size}' in cfg_lower:
            detected_size = size
            break
    
    if not detected_size:
        model_pt_lower = str(model_pt).lower()
        for size in model_sizes:
            if f'yolov9-{size}' in model_pt_lower or f'yolov9{size}' in model_pt_lower:
                detected_size = size
                break

    if not os.path.isfile(cfg):
        logger.warning(f'{colorstr("Model Exporter: ")}Config file not found: {cfg}')
        
        cfg_alternatives = []
        
        if detected_size:
            cfg_alternatives.append(CFG_PATH / 'yolov9' / f'yolov9-{detected_size}-converted.yaml')
            cfg_alternatives.append(CFG_PATH / 'yolov9' / f'yolov9-{detected_size}.yaml')
        
        for size in model_sizes:
            cfg_alternatives.append(CFG_PATH / 'yolov9' / f'yolov9-{size}-converted.yaml')
        
        for size in model_sizes:
            cfg_alternatives.append(CFG_PATH / 'yolov9' / f'yolov9-{size}.yaml')
        
        seen = set()
        unique_alternatives = []
        for alt in cfg_alternatives:
            if alt not in seen:
                seen.add(alt)
                unique_alternatives.append(alt)
        
        for alt_cfg in unique_alternatives:
            if alt_cfg.exists():
                cfg = str(alt_cfg)
                logger.info(f'{colorstr("Model Exporter: ")}Using config for YOLOv9-{detected_size or "unknown"}: {cfg}')
                break
        else:
            logger.warning(f'{colorstr("Model Exporter: ")}No suitable config found, returning original model')
            return model_pt
    
    try:
        model = DetectionModel(cfg, ch=3, nc=80, anchors=3).to(device)
        _ = model.eval()

        ckpt = torch.load(model_pt, map_location='cpu')
        
        if 'model' in ckpt and hasattr(ckpt['model'], 'names'):
            model.names = ckpt['model'].names
            model.nc = ckpt['model'].nc
        elif 'names' in ckpt:
            model.names = ckpt['names']
            model.nc = len(ckpt['names'])

        cfg_name = os.path.basename(cfg).lower()
        
        if detected_size:
            if detected_size in ['t', 's']:
                model = convert_small_model(model, ckpt)
                logger.info(f'{colorstr("Model Exporter: ")}Converting YOLOv9-{detected_size.upper()} (small model)')
            elif detected_size in ['m', 'c']:
                model = convert_medium_model(model, ckpt)
                logger.info(f'{colorstr("Model Exporter: ")}Converting YOLOv9-{detected_size.upper()} (medium model)')
            else:  # 'e' or 'supernet'
                model = convert_large_model(model, ckpt)
                logger.info(f'{colorstr("Model Exporter: ")}Converting YOLOv9-{detected_size.upper()} (large model)')
        else:
            if 'yolov9-t' in cfg_name or 'yolov9-s' in cfg_name:
                model = convert_small_model(model, ckpt)
            elif 'yolov9-m' in cfg_name or 'yolov9-c' in cfg_name:
                model = convert_medium_model(model, ckpt)
            else:
                model = convert_large_model(model, ckpt)

        reparamed_model = {
            'model': model.float(),
            'optimizer': None,
            'best_fitness': None,
            'epoch': -1,
            'ema': None,
            'updates': None,
            'training_results': None,
        }
        
        f_path = str(model_pt).replace('.pt', '_converted.pt')
        torch.save(reparamed_model, f_path)
        mb = os.path.getsize(f_path) / 1E6
        logger.info(f'{colorstr("Model Exporter: ")}Reparametered as {f_path}({mb:.1f}MB)')
        return f_path
        
    except Exception as e:
        logger.error(f'{colorstr("Model Exporter: ")}Conversion failed: {e}')
        logger.error(f'Traceback: {traceback.format_exc()}')
        return model_pt