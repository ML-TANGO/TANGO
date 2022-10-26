'''
NAS controllor
'''

# import time
import torch.onnx

from .enas import ENAS
from .eval import fine_tune
from ..utils.accelerate import check_amp, check_train_batch_size


def train(
        train_loader,
        val_loader,
        base_model,
        supernet,
        nc,
        names,
        max_latency,
        pop_size,
        niter,
        device):
    '''
    NAS controllor
    '''

    # ENAS ######  --> final net
    enas = ENAS(val_loader, 
                base_model, 
                supernet, 
                device, 
                nc, 
                names,
                max_latency,
                pop_size,
                niter)

    _, best_net = enas.run_evolution_search()

    amp = check_amp(best_net, final=True)  # check AMP
    best_net = fine_tune(train_loader, best_net, amp)
    best_net.eval()

    return best_net  
