import copy
import random
from tqdm import tqdm, trange
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ArchManager:
    def __init__(self, depth_list):
        if not isinstance(depth_list, list) or not all(isinstance(x, list) and x for x in depth_list):
            raise ValueError(f"[ArchManager] invalid depth_list: {depth_list}")
        self.depth_list = depth_list[:]
        self.num_blocks = len(self.depth_list)
        self.sample_set = set()

    def _choices_for(self, i):
        return self.depth_list[i]

    def random_sample(self):
        while True:
            d = [random.choice(self._choices_for(i)) for i in range(self.num_blocks)]
            key = tuple(d)
            if key not in self.sample_set:
                self.sample_set.add(key)
                sample = {"d": d}
                logger.info(sample)
                return sample

    def random_resample_depth(self, sample, i):
        assert 0 <= i < self.num_blocks
        sample["d"][i] = random.choice(self._choices_for(i))


class EvolutionFinder:

    valid_constraint_range = {
        "galaxy22": [150, 5000],
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
        if self.constraint_type not in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()

        self.efficiency_constraint = efficiency_constraint
        if not (
            self.valid_constraint_range[self.constraint_type][0]
            <= self.efficiency_constraint
            <= self.valid_constraint_range[self.constraint_type][1]
        ):
            self.invite_reset_constraint()

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        supernet = getattr(self.accuracy_predictor, "supernet", None)
        depth_list = getattr(supernet, "depth_list", None)
        if not depth_list:
            raise ValueError("[EvolutionFinder] accuracy_predictor.supernet.depth_list 가 없습니다.")
        self.arch_manager = ArchManager(depth_list)
        self.num_blocks = self.arch_manager.num_blocks

        self.population_size = kwargs.get("population_size", 1)
        self.num_generations = kwargs.get("num_generations", 500)
        self.parent_ratio = kwargs.get("parent_ratio", 1.0)
        self.mutate_prob = kwargs.get("mutate_prob", 0.1)
        self.mutation_ratio = kwargs.get("mutation_ratio", 0.5)
        self.max_time_budget = kwargs.get("max_time_budget", 1)

    def invite_reset_constraint_type(self):
        logger.warning(
            "Invalid constraint type! Please input one of: %s",
            list(self.valid_constraint_range.keys()),
        )
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            logger.warning(
                "Invalid constraint type! Please input one of: %s",
                list(self.valid_constraint_range.keys()),
            )
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        lo, hi = self.valid_constraint_range[self.constraint_type]
        logger.warning(
            "Invalid constraint_value! Please input an integer in interval: [%d, %d]!",
            lo, hi
        )
        new_cons = input()
        while (not new_cons.isdigit()) or (int(new_cons) > hi) or (int(new_cons) < lo):
            logger.warning(
                "Invalid constraint_value! Please input an integer in interval: [%d, %d]!",
                lo, hi
            )
            new_cons = input()
        self.efficiency_constraint = int(new_cons)

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def _ensure_valid_sample(self, sample):
        """
        - 길이가 모자라면 해당 위치의 허용 후보에서 랜덤으로 채움
        - 길이가 길면 잘라냄
        - 값이 허용 후보에 없으면 가까운 허용값으로 교체(간단화: 랜덤 치환)
        """
        if "d" not in sample or not isinstance(sample["d"], list):
            raise ValueError(f"[EvolutionFinder] invalid sample: {sample}")
        d = sample["d"]

        if len(d) < self.num_blocks:
            for i in range(len(d), self.num_blocks):
                d.append(random.choice(self.arch_manager._choices_for(i)))
        elif len(d) > self.num_blocks:
            del d[self.num_blocks:]

        for i in range(self.num_blocks):
            choices = self.arch_manager._choices_for(i)
            if d[i] not in choices:
                d[i] = random.choice(choices)
        sample["d"] = d
        return sample

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            sample = self._ensure_valid_sample(sample)
            efficiency = self.efficiency_predictor.predict_efficiency(sample)
            if efficiency <= constraint:
                return sample, efficiency

    def mutate_sample(self, sample):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample)
            new_sample = self._ensure_valid_sample(new_sample)

            for i in range(self.num_blocks):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)

            new_sample = self._ensure_valid_sample(new_sample)
            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            s1 = self._ensure_valid_sample(copy.deepcopy(sample1))
            s2 = self._ensure_valid_sample(copy.deepcopy(sample2))
            new_sample = {"d": [random.choice([s1["d"][i], s2["d"][i]]) for i in range(self.num_blocks)]}
            new_sample = self._ensure_valid_sample(new_sample)

            efficiency = self.efficiency_predictor.predict_efficiency(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency

    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))

        best_valids = [-100]
        population = []
        child_pool = []
        best_info = None

        for _ in trange(population_size, desc="Generate random population..."):
            sample, efficiency = self.random_sample()
            sample = self._ensure_valid_sample(sample)
            subnet, acc = self.accuracy_predictor.predict_accuracy_once(sample)
            population.append((acc, sample, float(efficiency), subnet))

        if verbose:
            for i, (a, s, e, n) in enumerate(population):
                logger.info(f"[{i}] acc={a:.4f}, config={s['d']}, flops={e:.1f}M, model={n}")
            logger.info("Start Evolution...")

        for iter in tqdm(
            range(max_time_budget),
            desc=f"Searching with {self.constraint_type} constraint ({self.efficiency_constraint})"
        ):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                logger.info("Iter: %d Acc: %.6f", iter + 1, parents[0][0])

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
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(float(efficiency))

            for _ in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(float(efficiency))

            for i in trange(population_size, desc=f"[{iter+1}|{max_time_budget}] Mutate and Crossover..."):
                s = self._ensure_valid_sample(child_pool[i])
                subnet, acc = self.accuracy_predictor.predict_accuracy_once(s)
                population.append((acc, s, efficiency_pool[i], subnet))

        return best_valids, best_info
