'''
The currently written version uses real evluation MAP calculated for test dataset.
Accuracy predictor will be added to increase the search efficiency.
'''

import os
import yaml
import torch
import torch.distributed as dist
from copy import deepcopy
from finetune import finetune
from test import test  # import test.py to get mAP for each subnet
from tqdm import tqdm

from yolov7_utils.datasets import create_dataloader
from yolov7_utils.general import colorstr, check_img_size
from yolov7_utils.torch_utils import select_device


class AccuracyCalculator():
    def __init__(
        self, 
        opt,
        supernet,

    ):
        self.opt = opt
        self.supernet = supernet

        # Set DDP variables
        opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
        opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

        # Resume
        # if opt.resume:  # resume an interrupted run
        #     ckpt = opt.weights if opt.weights.endswith('.pt') else last(opt.weights + '*.pt')  # checkpoint path
        
        # DDP mode
        opt.total_batch_size = opt.batch_size
        self.device = select_device(opt.device, batch_size=opt.batch_size)
        if opt.local_rank != -1:
            assert torch.cuda.device_count() > opt.local_rank
            torch.cuda.set_device(opt.local_rank)
            self.device = torch.device('cuda', opt.local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
            assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
            opt.batch_size = opt.total_batch_size // opt.world_size
            
        # Hyperparameters
        with open(opt.hyp) as f:
            self.hyp = hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
            
        # load data yaml
        with open(opt.data, encoding="UTF-8") as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        self.is_coco = opt.data.endswith('coco.yaml') or opt.data.endswith('coco128.yaml')
        
        # Image sizes
        gs = max(int(supernet.stride.max()), 32)  # grid size (max stride)
        imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]
        self.imgsz_test = imgsz_test
        # prepare test dataset
        test_path = data_dict['val']
        self.testloader = create_dataloader(test_path, imgsz_test, opt.batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]
        
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
        return finetune(subnet, self.hyp, self.opt, self.device)
        # raise NotImplementedError
        

    def predict_accuracy(self, sample_list):
        acc_list = []
        # sample_list: list of subnets
        for sample in sample_list:
            self.supernet.set_active_subnet(sample['d'])
            
            # Calculate mAP
            results, _, _ = test(self.opt.data,
                                batch_size=self.opt.batch_size * 2,
                                imgsz=self.imgsz_test,
                                conf_thres=0.001,
                                iou_thres=0.7,
                                model=self.supernet,
                                single_cls=self.opt.single_cls,
                                dataloader=self.testloader,
                                # save_dir=save_dir,
                                save_json=False,
                                plots=False,
                                is_coco=self.is_coco,
                                v5_metric=self.opt.v5_metric)
            
            # mp, mr, map50, map, avg_loss = results
            map= results[3]
            acc_list.append(map)

        return acc_list
    
    def predict_accuracy_once(self, sample):
        # activate the subnet
        self.supernet.set_active_subnet(sample['d'])
        # TODO : check the speed and memory about two implementations 1) get_active_subnet() 2) set_active_subnet() and deepcopy
        # 1) get_active_subnet()
        subnet = self.supernet.get_active_subnet()
        # 2) set_active_subnet() and deepcopy
        # subnet = deepcopy(self.supernet)
        
        # finetune the subnet
        subnet, finetune_results = self.finetune_subnet(subnet)
        
        # Calculate mAP
        results, _, _ = test(self.opt.data,
                            batch_size=self.opt.batch_size * 2,
                            imgsz=self.imgsz_test,
                            conf_thres=0.001,
                            iou_thres=0.7,
                            model=subnet,
                            single_cls=self.opt.single_cls,
                            dataloader=self.testloader,
                            # save_dir=save_dir,
                            save_json=False,
                            plots=False,
                            is_coco=self.is_coco,
                            v5_metric=self.opt.v5_metric)
            
        # mp, mr, map50, map, avg_loss = results
        map = results[3]
        return map
    
    