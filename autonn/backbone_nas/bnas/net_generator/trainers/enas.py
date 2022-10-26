'''
Evolutionary Algorithm-based NAS
'''

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

# from trainers.search_config import config
from .nas_utils import sortNondominated, CrowdingDist
from ..latency_lookup_table import LatencyTable
from ..models.common import Conv, C3
from .eval import cal_metrics
from ..utils.pytorch_visual import Pytorch_Visual

class ArchManager:
    '''
    Arch manager
    '''
    def __init__(self):
        self.num_blocks = 20
        self.num_stages = 5
        self.kernel_sizes = [3, 5, 7]
        self.expand_ratios = [3, 4, 6]
        self.depths = [2, 3, 4]
        self.resolutions = [224]  # not used 160, 176, 192, 208

    def random_sample(self):
        '''
        random sampling
        '''
        sample = {}
        d = []
        e = []
        ks = []
        for _ in range(self.num_stages):
            d.append(random.choice(self.depths))

        for _ in range(self.num_blocks):
            e.append(random.choice(self.expand_ratios))
            ks.append(random.choice(self.kernel_sizes))

        sample = {
            # "wid": None,
            "ks": ks,
            "e": e,
            "d": d,
            "r": [random.choice(self.resolutions)],
        }

        return sample

    def random_resample(self, sample, i):
        '''
        resampling
        '''
        assert self.num_blocks > i >= 0
        sample["ks"][i] = random.choice(self.kernel_sizes)
        sample["e"][i] = random.choice(self.expand_ratios)

    def random_resample_depth(self, sample, i):
        '''
        resample depth
        '''
        assert self.num_stages > i >= 0
        sample["d"][i] = random.choice(self.depths)

    def random_resample_resolution(self, sample):
        '''
        resample resolution
        '''
        sample["r"][0] = random.choice(self.resolutions)


class ENAS:
    valid_constraint_range = {
        "flops": [150, 600],
        "note10": [15, 60],
    }

    def __init__(
        self,
        val_loader, 
        base_model, 
        supernet, 
        device, 
        nc, 
        names,
        efficiency_constraint,
        pop_size,
        niter,
        constraint_type='note10',
        efficiency_predictor=LatencyTable(device='note10'),
        accuracy_predictor=None,
        map_evaluator=None,
        **kwargs
    ):

        self.supernet = supernet  # SuperNet
        self.head = base_model
        self.nc = nc
        self.names = names
        self.val_loader = val_loader
        self.device = device

        self.constraint_type = constraint_type
        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_constraint = efficiency_constraint
        if not (
            efficiency_constraint <= self.valid_constraint_range[constraint_type][1]
            and efficiency_constraint >= self.valid_constraint_range[constraint_type][0]
        ):
            self.invite_reset_constraint()

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager()
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        self.population_size = pop_size
        self.max_time_budget = niter
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        self.tournament_size = max(2, int(self.population_size * 0.5)) # max(2, 20*0.5)

    def invite_reset_constraint_type(self):
        print(
            "Invalid constraint type! Please input one of:",
            list(self.valid_constraint_range.keys()),
        )
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print(
                "Invalid constraint type! Please input one of:",
                list(self.valid_constraint_range.keys()),
            )
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        print(
            "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
            % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1],
            )
        )

        new_cons = input()
        while (
            (not new_cons.isdigit())
            or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1])
            or (int(new_cons) < self.valid_constraint_range[self.constraint_type][0])
        ):
            print(
                "Invalid constraint_value! Please input an integer in interval: [%d, %d]!"
                % (
                    self.valid_constraint_range[self.constraint_type][0],
                    self.valid_constraint_range[self.constraint_type][1],
                )
            )
            new_cons = input()
        new_cons = int(new_cons)
        self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= constraint:
                return sample, efficiency

    def mutate_sample(self, sample):
        constraint = self.efficiency_constraint
        while True:
            new_sample = deepcopy(sample)

            if random.random() < self.mutate_prob:
                self.arch_manager.random_resample_resolution(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample(new_sample, i)

            for i in range(self.num_stages):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            new_sample = deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice(
                        [sample1[key][i], sample2[key][i]]
                    )

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def run_evolution_search(self, verbose=True):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None

        

        # log 출력 형식
        #formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # log 출력
        #stream_handler = logging.StreamHandler()
        #stream_handler.setFormatter(formatter)
        #logger.addHandler(stream_handler)

        # log를 파일에 출력
        #file_handler = logging.FileHandler('log')
        #file_handler.setFormatter(formatter)
        #logger.addHandler(file_handler)

        if verbose:
            print("Generate random population...")
        for _ in range(population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        # accs = self.accuracy_predictor.predict_accuracy(child_pool)
        accs = self._evaluate(child_pool)
        for i in range(population_size):
            population.append((accs[i], child_pool[i], efficiency_pool[i]))

        if verbose:
            print("Start Evolution...")
            #logger = logging.getLogger()
            #logger.setLevel(logging.INFO)
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="Searching with %s constraint (%s)"
            % (self.constraint_type, self.efficiency_constraint),
        ):
            eval_vec = []
            for idx in range(len(population)):
                eval_vec.append((population[idx][0], population[idx][2]))
            idx_list = self._sorting(eval_vec)
            population = [population[i] for i in idx_list]

            acc = population[0][0]
            if verbose:
                print("************************")
                print("Iter: {} Acc: {} Time: {}".format(iter + 1, population[0][0], population[0][2]))
                #k_list = population[0][1]['ks']
                #out_str = ' '.join(str(e) for e in k_list)
                #logger.info("kernel Info " + out_str)
                #logger.info("mAP " + str(population[0][0]))
                #logger.info("time " + str(population[0][2]))
                print("************************")
            # Save image
            # x = Variable(torch.rand(1, 3, 128, 256)).to(self.device).float() 
            # samp = SampledModel(self.head, self.supernet, population[0][1])
            
            #make_dot(samp(x)[0], params=dict(samp.named_parameters())).render("graph_{}".format(iter-1), format="png")
            # pc = Pytorch_Visual(samp, x)                
            # pc.visualization(f'graph_{iter+1}',format='jpg')

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = population[0]
            else:
                best_valids.append(best_valids[-1])

            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):
                idx, _ = self._tournament_selection()
                par_sample = population[idx][1]
                # Mutate
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):
                idx1, idx2 = self._tournament_selection()
                par_sample1 = population[idx1][1]
                par_sample2 = population[idx2][1]
                # Crossover
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            accs = self._evaluate(child_pool)
            for i in range(population_size):
                population.append((accs[i], child_pool[i], efficiency_pool[i]))

        _, net_config, latency = best_info
        b_net = SampledModel(self.head, self.supernet, net_config)
        return best_valids, b_net

    def _tournament_selection(self):
        indices = np.random.choice(self.population_size, self.tournament_size)
        indices = np.sort(indices)
        return indices[0], indices[1]


    def _sorting(self, eval_vec):
        fronts = sortNondominated(eval_vec)

        idx_list = []

        for i in range(len(fronts)):
            if len(fronts[i]) == 1:
                idx_list.append(fronts[i][0])
            elif len(fronts[i]) == 2:
                idx_list.append(fronts[i][0])
                idx_list.append(fronts[i][1])
            else:
                vec = [eval_vec[idx] for idx in fronts[i]]
                distances = CrowdingDist(vec)
                eval_indices = list(np.argsort(np.array(distances))[::-1])
                for j in range(len(fronts[i])):
                    idx_list.append(fronts[i][eval_indices[j]])

            if len(idx_list) >= self.population_size:
                idx_list = idx_list[0:self.population_size]
                break

        return idx_list


    def _evaluate(self, population):
        """
        _evaluate
        """
        accs = []
        for sample in population:
            net = SampledModel(self.head, self.supernet, sample)
            mAP = cal_metrics(self.val_loader, net, self.nc, self.names)
            accs.append(mAP)

        return accs

class SampledModel(nn.Module):
    '''
    model sampling
    '''

    def __init__(self, base_model, supernet, arch):
        super().__init__()
        self.arch = arch
        self.head = base_model
        self.stride = self.head.stride
        self.supernet = supernet
        # arch format {'ks':[], 'e':[], 'd':[]}
        self.supernet.set_active_subnet(**arch)
        self.supernet.set_backbone_fpn(returned_layers=[1, 2, 3])

    def forward(self, im):
        # return outputs, outputs.keys()
        o, k = self.supernet.forward_fpn(im) 
        multiScaleFs = [o[k[0]], o[k[1]], o[k[2]]]
        x = o[k[2]]

        for m in self.head:
            if m.f != -1:
                if m.f[1]>7:
                    x = [x if j == -1 else multiScaleFs[j-7] for j in m.f]
                elif m.f[1]==6:
                    x = [x, multiScaleFs[1]]
                elif m.f[1]==4:
                    x = [x, multiScaleFs[0]]

            if isinstance(m, Conv):
                in_C = m.conv.in_channels
                pad = torch.zeros(x.shape[0], in_C, x.shape[-2], x.shape[-1]).cuda()
                pad[:, :x.shape[1], ...] = x
                x = pad
            elif isinstance(m, C3):
                in_C = m.cv1.conv.in_channels
                pad = torch.zeros(x.shape[0], in_C, x.shape[-2], x.shape[-1]).cuda()
                pad[:, :x.shape[1], ...] = x
                x = pad
            x = m(x)
            multiScaleFs.append(x)
        return x
