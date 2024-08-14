'''
The currently written version uses real evluation MAP calculated for test dataset.
Accuracy predictor will be added to increase the search efficiency.
[TEANCE] it uses the 'supernet' as pre-trained weights (deep-copy)
         and fine-tunes a 'subnet' to get its accuracy.
         thus, no need to all the settings for test dataset and their evaluation.
         it has already performed at select.py and train.py for the supernet
         it makes this class 'AccuracyCalculator' very simple.
'''

import os
import sys
import yaml
import logging
import torch
import torch.distributed as dist
from copy import deepcopy
from tqdm import tqdm

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # to run '$ python *.py' files in subdirectories
# sys.path.append('.../')  # to run '$ python *.py' files in subdirectories
from tango.main import test # import test.py to get mAP after each epoch
from tango.main import finetune
from tango.utils.datasets import create_dataloader
from tango.utils.general import colorstr, check_img_size
from tango.utils.torch_utils import select_device
from tango.common.models.experimental import attempt_load

logger = logging.getLogger(__name__)


class AccuracyCalculator():
    def __init__(
        self,
        proj_info,
        hyp,
        opt,
        data_dict,
        supernet,
    ):
        self.opt = opt
        self.supernet = supernet

        self.proj_info = proj_info
        self.userid = proj_info['userid']
        self.project_id = proj_info['project_id']

        # Set DDP variables : [TENACE] already done this at select.py (line 169-170)
        # opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        # opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        # Resume
        # if opt.resume:  # resume an interrupted run
        #     ckpt = opt.weights if opt.weights.endswith('.pt') else last(opt.weights + '*.pt')  # checkpoint path
        
        # DDP mode : [TENACE] already done this at train.py (line 212-221)
        # opt.total_batch_size = opt.batch_size
        # self.device = select_device(opt.device, batch_size=opt.batch_size)
        # if opt.local_rank != -1:
        #     assert torch.cuda.device_count() > opt.local_rank
        #     torch.cuda.set_device(opt.local_rank)
        #     self.device = torch.device('cuda', opt.local_rank)
        #     dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        #     assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        #     opt.batch_size = opt.total_batch_size // opt.world_size
            
        # Hyperparameters
        # with open(opt.hyp) as f:
        #     self.hyp = hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        # self.hyp = opt.hyp
        self.hyp = hyp
            
        # load data yaml : [TENACE] already done this at select.py (line 134-135)
        # with open(opt.data, encoding="UTF-8") as f:
        #     data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.data_dict = data_dict
        # test_path = self.data_dict['val']
        # self.is_coco = True if self.data_dict['dataset_name'] == 'coco' and 'coco' in test_path else False
        
        # Image sizes: [TENACE] train.py (line 191-194)
        # gs = max(int(supernet.stride.max()), 32)  # grid size (max stride)
        # imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
        # self.imgsz_test = imgsz_test

        # prepare test dataset: [TENACE] train.py (line 434-449)
        # self.testloader = create_dataloader(self.userid,
        #                                     self.project_id,
        #                                     test_path,
        #                                     self.imgsz_test,
        #                                     opt.batch_size, # * 2,
        #                                     gs,
        #                                     opt,
        #                                     hyp=self.hyp,
        #                                     cache=opt.cache_images and not opt.notest,
        #                                     rect=True,
        #                                     rank=-1,
        #                                     world_size=opt.world_size,
        #                                     workers=opt.workers,
        #                                     pad=0.5,
        #                                     prefix='val')[0]
        
    # TODO : add finetune function
    def finetune_subnet(self, subnet):
        """ Finetune subnet with given hyperparameters and optimizer 
            Args:
            -----------
            subnet: YOLOModel
                a YOLOModel instance to finetune

            Attributes:
            -----------
            model: nn.Sequential
                a sequence of nn.modules, i.e. YOLOSuperNet modules 
            save: list
                indice of jumping points to use for forward pass
            depth_list: list of int
                list of depth for each ELANBlock
            runtime_depth: list of int
                list of depth for each ELANBlock of subnetwork, but initialized int at first
        """  
        
        # finetune the subnet
        # subnet, maps = finetune(subnet, self.hyp, self.opt, self.device)
        return finetune.finetune(self.proj_info, subnet, self.hyp, self.opt, self.data_dict, tb_writer=None)
        # raise NotImplementedError

    # def predict_accuracy(self, sample_list):
    #     acc_list = []
    #     # sample_list: list of subnets
    #     for sample in sample_list:
    #         self.supernet.set_active_subnet(sample['d'])
            
    #         # Calculate mAP
    #         results, _, _ = test.test(self.proj_info,
    #                                   self.data_dict, #self.opt.data,
    #                                   batch_size=self.opt.batch_size, # * 2,
    #                                   imgsz=self.imgsz_test,
    #                                   conf_thres=0.001,
    #                                   iou_thres=0.7,
    #                                   model=self.supernet.half(),   # change
    #                                   single_cls=self.opt.single_cls,
    #                                   dataloader=self.testloader,
    #                                   # save_dir=save_dir,
    #                                   save_json=False,
    #                                   plots=False,
    #                                   is_coco=self.is_coco,
    #                                   v5_metric=self.opt.v5_metric)
            
    #         # mp, mr, map50, map, avg_loss = results
    #         map= results[3]
    #         acc_list.append(map)

    #     return acc_list
    
    def predict_accuracy_once(self, sample):
        # activate the subnet
        self.supernet.set_active_subnet(sample['d'])
        # TODO : check the speed and memory about two implementations 1) get_active_subnet() 2) set_active_subnet() and deepcopy
        # 1) get_active_subnet()
        subnet = self.supernet.get_active_subnet() # subnet : nn.Module
        # 2) set_active_subnet() and deepcopy
        # subnet = deepcopy(self.supernet)

        # finetune the subnet
        # out subnet : str (path/to/subnet.pt) <- in subnet : nn.Module (Model)
        subnet, finetune_results = self.finetune_subnet(subnet)

        # Calculate mAP
        # results, _, _ = test.test(self.opt.data,
        #                           batch_size=self.opt.batch_size * 2,
        #                           imgsz=self.imgsz_test,
        #                           conf_thres=0.001,
        #                           iou_thres=0.7,
        #                           model=subnet.half(),
        #                           single_cls=self.opt.single_cls,
        #                           dataloader=self.testloader,
        #                           # save_dir=save_dir,
        #                           save_json=False,
        #                           plots=False,
        #                           is_coco=self.is_coco,
        #                           v5_metric=self.opt.v5_metric)
            
        # mp, mr, map50, map, avg_loss = results
        map = finetune_results[3]
        return subnet, map
