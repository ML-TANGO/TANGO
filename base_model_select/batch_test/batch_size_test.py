from .models.yolo import Model
from .utils.autobatch import get_batch_size_for_gpu


def run_batch_test(cfg_yaml, hyp_yaml):
    with open(hyp_yaml, 'r') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    model = Model(cfg_yaml, ch=3, nc=80, anchors=hyp.get('anchors'))
    batch_size = int(get_batch_size_for_gpu(model, max(opt.img_size), amp=True) * 0.8)
    # 0.8 is multiplied by batch size to prevent cuda memory error due to a memory leak of yolov7
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return batch_size
