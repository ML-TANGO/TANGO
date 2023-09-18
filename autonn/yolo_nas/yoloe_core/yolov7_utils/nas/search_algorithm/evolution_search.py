import copy
import random
from tqdm import tqdm, trange
import numpy as np


class ArchManager:
    def __init__(self):
        self.num_blocks = [4, 4]
        self.depths = [[1,2,3], [1,2,3,4,5]]
        # self.resolutions = [160, 176, 192, 208, 224]

    def random_sample(self):
        sample = {}
        d = []
        for i in range(self.num_blocks[0]):
            d.append(random.choice(self.depths[0]))
        for i in range(self.num_blocks[1]):
            d.append(random.choice(self.depths[1]))

        sample = {
            "d": d,
        }

        return sample

    def random_resample_depth(self, sample, i):
        assert i >= 0
        if i < self.num_blocks[0]:
            sample["d"][i] = random.choice(self.depths[0])
        else:
            sample["d"][i] = random.choice(self.depths[1])

    # def random_resample_resolution(self, sample):
    #     sample["r"][0] = random.choice(self.resolutions)
    
    
class EvolutionFinder:

    valid_constraint_range = {
        "galaxy10": [150, 5000],
        "note10": [15, 60],
    }
    def __init__(
        self,
        constraint_type,
        efficiency_constraint,
        efficiency_predictor,
        accuracy_predictor,
        **kwargs
    ):
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
        self.num_blocks = self.arch_manager.num_blocks # [4, 4]

        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        self.population_size = kwargs.get("population_size", 5)
        self.max_time_budget = kwargs.get("max_time_budget", 5)
        self.parent_ratio = kwargs.get("parent_ratio", 0.25)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        
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
            new_sample = copy.deepcopy(sample)

            for i in range(sum(self.num_blocks)):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample1)
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

    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))
        constraint = self.efficiency_constraint

        best_valids = [-100]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        best_info = None

        for _ in trange(population_size, desc="Generate random population..."):
            sample, efficiency = self.random_sample()
            acc = self.accuracy_predictor.predict_accuracy_once(sample)
            population.append((acc, sample, efficiency))

        if verbose:
            print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(
            range(max_time_budget),
            desc="Searching with %s constraint (%s)"
            % (self.constraint_type, self.efficiency_constraint),
        ):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print("Iter: {} Acc: {}".format(iter - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                # Mutate
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                # Crossover
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in trange(population_size, desc=f"[{iter+1}|{max_time_budget}] Mutate and Crossover..."):
                acc = self.accuracy_predictor.predict_accuracy_once(child_pool[i])                
                population.append((acc, child_pool[i], efficiency_pool[i]))

        return best_valids, best_info