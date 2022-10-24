'''
NAS controllor
'''

# import time

from .enas import ENAS
from .eval import fine_tune
from ..utils.accelerate import check_amp


def train(
        _,
        val_loader,
        base_model,
        supernet,
        _nc,
        names,
        device):
    '''
    NAS controllor
    '''

    # ENAS ######  --> final net
    enas = ENAS(val_loader, base_model, supernet, device, _nc, names)

    # enas.initialize()
    # enas.search()
    # best_net, _ = enas.get_best()
    # st = time.time()
    _, best_net = enas.run_evolution_search()
    # ed = time.time()

    amp = check_amp(best_net)
    best_net = fine_tune(val_loader, best_net, amp)

    return best_net.eval()  # final net
