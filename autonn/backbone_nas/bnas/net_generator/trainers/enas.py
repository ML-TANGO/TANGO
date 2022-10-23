'''
Evolutionary Algorithm-based NAS
'''

import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# from trainers.search_config import config
# from trainers.nas_utils import random_can, sortNondominated, CrowdingDist
from ..latency_lookup_table import LatencyTable
from ..models.common import Conv, C3
from .eval import cal_metrics


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
        _d = []
        _e = []
        _ks = []
        for _ in range(self.num_stages):
            _d.append(random.choice(self.depths))

        for _ in range(self.num_blocks):
            _e.append(random.choice(self.expand_ratios))
            _ks.append(random.choice(self.kernel_sizes))

        sample = {
            # "wid": None,
            "ks": _ks,
            "e": _e,
            "d": _d,
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
    '''
    ENAS class
    '''
    valid_constraint_range = {
        "flops": [150, 600],
        "note10": [15, 60],
    }

    def __init__(
            self,
            val_loader,
            base_model,
            supernet,
            # device,
            _nc,
            names,
            constraint_type='note10',
            efficiency_constraint=25,
            efficiency_predictor=LatencyTable(device='note10'),
            accuracy_predictor=None,
            # map_evaluator=None,
            **kwargs
    ):

        self.supernet = supernet  # SuperNet
        self.head = base_model
        self._nc = _nc
        self.names = names
        self.val_loader = val_loader

        self.constraint_type = constraint_type
        if constraint_type not in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_constraint = efficiency_constraint
        if not (
                self.valid_constraint_range[constraint_type][1] >=
                efficiency_constraint >=
                self.valid_constraint_range[constraint_type][0]
        ):
            self.invite_reset_constraint()

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager()
        self.num_blocks = self.arch_manager.num_blocks
        self.num_stages = self.arch_manager.num_stages

        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        # self.population_size = kwargs.get("population_size", 100)
        # self.max_time_budget = kwargs.get("max_time_budget", 500)
        # testing
        self.population_size = kwargs.get("population_size", 4)
        self.max_time_budget = kwargs.get("max_time_budget", 1)

        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

    def invite_reset_constraint_type(self):
        '''
        reset cons type
        '''
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
        '''
        reset cons
        '''
        print(
            "Invalid constraint_value!\
                Please input an integer in interval: [%d, %d]!"
            % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1],
            )
        )

        new_cons = input()
        while (
                (not new_cons.isdigit())
                or (int(new_cons) >
                    self.valid_constraint_range[self.constraint_type][1])
                or (int(new_cons) <
                    self.valid_constraint_range[self.constraint_type][0])
        ):
            print(
                "Invalid constraint_value! \
                    Please input an integer in interval: [%d, %d]!"
                % (
                    self.valid_constraint_range[self.constraint_type][0],
                    self.valid_constraint_range[self.constraint_type][1],
                )
            )
            new_cons = input()
        new_cons = int(new_cons)
        self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        '''
        set constraint
        '''
        self.efficiency_constraint = new_constraint

    def random_sample(self):
        '''
        random sampling
        '''
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= constraint:
                return sample, efficiency

    def mutate_sample(self, sample):
        '''
        mutation
        '''
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

            efficiency = self.efficiency_predictor.predict_efficiency(
                new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        '''
        crossover
        '''
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

            efficiency = self.efficiency_predictor.predict_efficiency(
                new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def run_evolution_search(self, verbose=False):
        """
        Run a single roll-out of regularized evolution
        to a fixed time budget.
        """
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        # constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None
        if verbose:
            print("Generate random population...")
        for _ in range(population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        # accs = self.accuracy_predictor.predict_accuracy(child_pool)
        accs = self._evaluate(child_pool)
        for i in range(population_size):
            population.append(
                (accs[i].item(), child_pool[i], efficiency_pool[i]))

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for i in tqdm(
                range(max_time_budget),
                desc="Searching with %s constraint (%s)"
                % (self.constraint_type, self.efficiency_constraint),
        ):
            parents = sorted(population, key=lambda x: x[0])[
                ::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print("Iter: {} Acc: {}".format(i - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for _ in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for _ in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample, efficiency = self.crossover_sample(
                    par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            # accs = self.accuracy_predictor.predict_accuracy(child_pool)
            accs = self._evaluate(child_pool)
            for j in range(population_size):
                population.append(
                    (accs[j].item(), child_pool[j], efficiency_pool[j]))

        _, net_config, _ = best_info
        b_net = SampledModel(self.head, self.supernet, net_config)
        return best_valids, b_net

    def _evaluate(self, population):
        """
        _evaluate
        """
        accs = []

        for sample in population:
            net = SampledModel(self.head, self.supernet, sample)

            _map = cal_metrics(self.val_loader, net, self._nc, self.names)
            accs.append(_map)

        return accs

# to be automated


class SampledModel(nn.Module):
    '''
    model sampling
    '''

    def __init__(self, base_model, supernet, arch):
        super().__init__()
        self.head = base_model
        self.supernet = supernet
        # arch format {'ks':[], 'e':[], 'd':[]}
        self.supernet.set_active_subnet(**arch)
        self.supernet.set_backbone_fpn(returned_layers=[1, 2, 3])

    def forward(self, img):
        '''
        forward
        '''
        # return outputs, outputs.keys()
        _o, _k = self.supernet.forward_fpn(img)
        multi_scale_fs = [_o[_k[0]], _o[_k[1]], _o[_k[2]]]
        _x = _o[_k[2]]

        with torch.cuda.amp.autocast(False):
            for _m in self.head:
                if _m.f != -1:
                    if _m.f[1] > 7:
                        _x = [_x
                              if j == -1
                              else multi_scale_fs[j-7]
                              for j in _m.f]
                    elif _m.f[1] == 6:
                        _x = [_x, multi_scale_fs[1]]
                    elif _m.f[1] == 4:
                        _x = [_x, multi_scale_fs[0]]

                if isinstance(_m, Conv):
                    _in_c = _m.conv.in_channels
                    pad = torch.zeros(
                        _x.shape[0], _in_c, _x.shape[-2], _x.shape[-1]).cuda()
                    pad[:, :_x.shape[1], ...] = _x
                    _x = pad
                elif isinstance(_m, C3):
                    _in_c = _m.cv1.conv.in_channels
                    pad = torch.zeros(
                        _x.shape[0], _in_c, _x.shape[-2], _x.shape[-1]).cuda()
                    pad[:, :_x.shape[1], ...] = _x
                    _x = pad
                _x = _m(_x)
                multi_scale_fs.append(_x)
            return _x


# class ENAS:
#     #def __init__(self, model, val_loader, device):
#     def __init__(self, val_loader, base_model,
#                   supernet, device, nc, names, **kwargs):
#         self.best_map = []
#         self.model_params = []
#         self.model_enc = []
#         self.eval_vec = []
#         self.arch_manager = ArchManager() # search space manager
#         self.search_space = config.SEARCH_SPACE # not used
#         self.bottleneck_num = config.bottleneck_num # not used
#         self.supernet = supernet  # SuperNet
#         self.head = base_model
#         self.nc = nc
#         self.names = names
#         self.val_loader = val_loader
#         self.device = device
#         self.pop_size = config.population_num # 20
#         self.tournament_size = \
#           max(2, int(self.pop_size * config.tournament_portion))
#           # max(2, 20*0.5)

#         self.mutate_prob = kwargs.get("mutate_prob", 0.1)
#         self.population_size = kwargs.get("population_size", 20)
#         self.max_time_budget = kwargs.get("max_time_budget", 500)
#         self.parent_ratio = kwargs.get("parent_ratio", 0.25)
#         self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)

#     def initialize(self):
#         self.candidates = random_can(self.pop_size, config.states)

#         for i in range(self.pop_size):
#             self.eval_vec.append(self._evaluate(i))
#         self._sorting()
#         self._set_best()
#         return self.candidates

#     def _tournament_selection(self):
#         indices = np.random.choice(self.pop_size, self.tournament_size)
#         indices = np.sort(indices)
#         return indices[0], indices[1]


#     def _sorting(self):
#         fronts = sortNondominated(self.eval_vec)

#         idx_list = []

#         for i in range(len(fronts)):
#             if len(fronts[i]) == 1:
#                 idx_list.append(fronts[i][0])
#             elif len(fronts[i]) == 2:
#                 idx_list.append(fronts[i][0])
#                 idx_list.append(fronts[i][1])
#             else:
#                 vec = [self.eval_vec[idx] for idx in fronts[i]]
#                 distances = CrowdingDist(vec)
#                 eval_indices = list(np.argsort(np.array(distances))[::-1])
#                 for j in range(len(fronts[i])):
#                     idx_list.append(fronts[i][eval_indices[j]])

#             if len(idx_list) >= self.pop_size:
#                 idx_list = idx_list[0:self.pop_size]
#                 break

#         self.eval_vec = [self.eval_vec[i] for i in idx_list]
#         self.candidates = [self.candidates[i] for i in idx_list]

#     def crossover(self, index1, index2):
#         p1, p2 = self.candidates[index1], self.candidates[index2]
#         child = tuple(choice([i,j]) for i,j in zip(p1,p2))
#         return child

#     def mutation(self, index):
#         rand = tuple(np.random.randint(i) for i in config.states)
#         p1 = self.candidates[index]
#         child = tuple(choice([i,j]) for i,j in zip(p1,rand))
#         return child

#     def search(self):
#         for gen in range(config.max_generations):
#             print('Generation = {}'.format(gen))
#             idx1, idx2 = self._tournament_selection()
#             new_cross = self.crossover(idx1, idx2)
#             mut_idx = np.random.choice(self.pop_size, 1)
#             new_mutat = self.mutation(mut_idx[0])
#             self.candidates.append(new_cross)
#             self.eval_vec.append(self._evaluate(-1))
#             self.candidates.append(new_mutat)
#             self.eval_vec.append(self._evaluate(-1))
#             self._sorting()
#             self.candidates = self.candidates[0:self.pop_size]
#             self.eval_vec = self.eval_vec[0:self.pop_size]

#             self._set_best()

#     def _set_best(self):
#         self.best_map = self.eval_vec[0]
#         self.best_chr = self.candidates[0]

#     def get_best(self):
#         b_net = sampled_model(self.head, self.supernet, self.best_chr)
#         return b_net, self.best_map

#     def _evaluate(self, index):
#         # val (mAP), lat
#         enc = self.candidates[index]

#         net = sampled_model(self.head, self.supernet, enc)

#         mAP = cal_metrics(self.val_loader, net, self.nc, self.names)

#         net_config = self.supernet.config(arch = enc)

#         # estimator = MBv2LatencyTable(url='mobile_lut.yaml')
#         estimator = LatencyTable(device='note10')
#         estimator.count_flops_given_config(net_config)
#         lat = estimator.predict_network_latency_given_config(net_config)
#         return (mAP, lat)
