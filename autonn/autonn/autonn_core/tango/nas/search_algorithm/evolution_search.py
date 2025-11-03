import copy
import random
from tqdm import tqdm, trange
import numpy as np
import logging

logger = logging.getLogger(__name__)

EPS = 1e-9


def to_float_scalar(x, default=0.0):
    try:
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return float(default)


def finite_or_default(x, default):
    x = to_float_scalar(x, default=default)
    if not np.isfinite(x):
        return float(default)
    return x


def clamp_min(x, minv=EPS):
    x = to_float_scalar(x, default=minv)
    if x < minv:
        return float(minv)
    return float(x)


class ArchManager:
    def __init__(self):
        self.num_blocks = [4, 8]
        self.depths = [[1, 2, 3], [1, 2, 3, 4]]
        self.sample_list = []

    def random_sample(self):
        sample = {}
        d = []

        def pick_sample():
            d.clear()
            for _ in range(self.num_blocks[0]):
                d.append(random.choice(self.depths[0]))
            for _ in range(self.num_blocks[1]):
                d.append(random.choice(self.depths[1]))

        pick_sample()

        tries = 0
        while d in self.sample_list and tries < 50:
            pick_sample()
            tries += 1

        self.sample_list.append(d)
        sample = {"d": d}
        logger.info(sample)
        return sample

    def random_resample_depth(self, sample, i):
        assert i >= 0
        if i < self.num_blocks[0]:
            sample["d"][i] = random.choice(self.depths[0])
        else:
            sample["d"][i] = random.choice(self.depths[1])


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
        if self.constraint_type not in self.valid_constraint_range:
            logger.warning(
                "Invalid constraint type '%s'. Falling back to 'galaxy22'.",
                self.constraint_type,
            )
            self.constraint_type = "galaxy22"

        low, high = self.valid_constraint_range[self.constraint_type]
        eff_c = to_float_scalar(efficiency_constraint, default=high)
        if not (low <= eff_c <= high):
            logger.warning(
                "Invalid constraint value %.3f for '%s'. Clamping to [%d, %d].",
                eff_c, self.constraint_type, low, high
            )
            eff_c = float(np.clip(eff_c, low, high))
        self.efficiency_constraint = eff_c

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor

        self.arch_manager = ArchManager()
        self.num_blocks = self.arch_manager.num_blocks  # [4, 8]

        self.population_size = max(int(kwargs.get("population_size", 16)), 1)
        self.num_generations = int(kwargs.get("num_generations", 500))
        self.parent_ratio = float(kwargs.get("parent_ratio", 0.5))
        self.mutate_prob = float(kwargs.get("mutate_prob", 0.1))
        self.mutation_ratio = float(kwargs.get("mutation_ratio", 0.5))
        self.max_time_budget = int(kwargs.get("max_time_budget", 50))

        self.constraint_relax_gamma = float(kwargs.get("constraint_relax_gamma", 1.05))
        self.max_trials_per_pick = int(kwargs.get("max_trials_per_pick", 200))
        
    def set_efficiency_constraint(self, new_constraint, clamp: bool = True):
        low, high = self.valid_constraint_range.get(self.constraint_type, (None, None))
        if low is None:
            logger.warning("Unknown constraint_type '%s'. Keeping previous constraint=%.3f",
                           self.constraint_type, self.efficiency_constraint)
            return float(self.efficiency_constraint)

        val = to_float_scalar(new_constraint, default=self.efficiency_constraint)
        if clamp:
            val = float(np.clip(val, low, high))
        else:
            if not (low <= val <= high):
                logger.warning(
                    "Ignored out-of-range constraint %.6f for '%s' (valid: [%d, %d]). "
                    "Keeping previous constraint=%.3f",
                    val, self.constraint_type, low, high, self.efficiency_constraint
                )
                return float(self.efficiency_constraint)

        if val <= 0.0:
            val = max(val, EPS)

        self.efficiency_constraint = float(val)
        logger.info("Set efficiency_constraint=%.6f for type='%s'", self.efficiency_constraint, self.constraint_type)
        return self.efficiency_constraint

    def set_constraint_type(self, new_type: str, keep_constraint: bool = False):
        if new_type not in self.valid_constraint_range:
            logger.warning("Invalid constraint type '%s'. Keeping '%s'.",
                           new_type, self.constraint_type)
            return self.constraint_type

        self.constraint_type = new_type
        low, high = self.valid_constraint_range[self.constraint_type]
        if keep_constraint:
            self.efficiency_constraint = float(np.clip(
                to_float_scalar(self.efficiency_constraint, default=high), low, high
            ))
        else:
            self.efficiency_constraint = float(high)

        logger.info("Set constraint_type='%s', efficiency_constraint=%.6f",
                    self.constraint_type, self.efficiency_constraint)
        return self.constraint_type
    
    def predict_efficiency_safe(self, sample):
        try:
            val = self.efficiency_predictor.predict_efficiency(sample)
        except ZeroDivisionError:
            logger.warning("Efficiency predictor raised ZeroDivisionError. Penalizing.")
            return float(np.inf)
        except Exception as e:
            logger.warning("Efficiency predictor exception: %s. Penalizing.", e)
            return float(np.inf)

        val = finite_or_default(val, default=np.inf)
        # 분모가 되는 경로 방지: 0 또는 음수면 매우 큰 값으로
        if val <= 0.0:
            return float(np.inf)
        return float(val)

    def predict_accuracy_safe(self, sample):
        try:
            subnet, acc = self.accuracy_predictor.predict_accuracy_once(sample)
        except ZeroDivisionError:
            logger.warning("Accuracy predictor raised ZeroDivisionError. Setting acc=0.")
            return None, 0.0
        except Exception as e:
            logger.warning("Accuracy predictor exception: %s. Setting acc=0.", e)
            return None, 0.0

        acc = finite_or_default(acc, default=0.0)
        return subnet, float(acc)

    def _accept_under_constraint(self, efficiency, constraint):
        eff = finite_or_default(efficiency, default=np.inf)
        con = to_float_scalar(constraint, default=np.inf)
        return eff <= con

    def random_sample(self):
        constraint = float(self.efficiency_constraint)
        best = None  # (eff, sample)
        for t in range(self.max_trials_per_pick):
            sample = self.arch_manager.random_sample()
            efficiency = self.predict_efficiency_safe(sample)
            if self._accept_under_constraint(efficiency, constraint):
                return sample, efficiency
            if best is None or efficiency < best[0]:
                best = (efficiency, sample)

        logger.warning(
            "random_sample: could not meet constraint=%.3f in %d trials. "
            "Relaxing by x%.3f (best eff=%.3f).",
            constraint, self.max_trials_per_pick, self.constraint_relax_gamma, best[0]
        )
        self.efficiency_constraint *= self.constraint_relax_gamma
        return best[1], best[0]

    def mutate_sample(self, sample):
        constraint = float(self.efficiency_constraint)
        best = None
        for _ in range(self.max_trials_per_pick):
            new_sample = copy.deepcopy(sample)
            for i in range(sum(self.num_blocks)):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample_depth(new_sample, i)
            efficiency = self.predict_efficiency_safe(new_sample)
            if self._accept_under_constraint(efficiency, constraint):
                return new_sample, efficiency
            if best is None or efficiency < best[0]:
                best = (efficiency, new_sample)

        logger.warning(
            "mutate_sample: constraint unmet; relaxing to x%.3f (best eff=%.3f).",
            self.constraint_relax_gamma, best[0]
        )
        self.efficiency_constraint *= self.constraint_relax_gamma
        return best[1], best[0]

    def crossover_sample(self, sample1, sample2):
        constraint = float(self.efficiency_constraint)
        best = None
        for _ in range(self.max_trials_per_pick):
            new_sample = copy.deepcopy(sample1)
            for key in new_sample.keys():
                if not isinstance(new_sample[key], list):
                    continue
                for i in range(len(new_sample[key])):
                    new_sample[key][i] = random.choice([sample1[key][i], sample2[key][i]])
            efficiency = self.predict_efficiency_safe(new_sample)
            if self._accept_under_constraint(efficiency, constraint):
                return new_sample, efficiency
            if best is None or efficiency < best[0]:
                best = (efficiency, new_sample)

        logger.warning(
            "crossover_sample: constraint unmet; relaxing to x%.3f (best eff=%.3f).",
            self.constraint_relax_gamma, best[0]
        )
        self.efficiency_constraint *= self.constraint_relax_gamma
        return best[1], best[0]

    def run_evolution_search(self, verbose=False):
        max_time_budget = int(self.max_time_budget)
        population_size = max(int(self.population_size), 1)
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        mutation_numbers = max(min(mutation_numbers, population_size), 0)
        parents_size = int(round(self.parent_ratio * population_size))
        parents_size = max(min(parents_size, population_size), 1)

        best_valids = [-100.0]
        population = []  # (acc, sample, efficiency, subnet) 튜플
        best_info = None

        logger.info("Generate random population...")
        # for _ in trange(population_size, desc="Generate random population..."):
        for x in range(population_size):
            logger.info(f'-----{x}/{population_size-1}')
            sample, efficiency = self.random_sample()
            subnet, acc = self.predict_accuracy_safe(sample)
            acc = finite_or_default(acc, default=0.0)
            population.append((acc, sample, clamp_min(efficiency), subnet))

        if verbose:
            for i, (a, s, e, n) in enumerate(population):
                logger.info(f"\n[{i}] acc={a:.6f}, config={s['d']}, eff={e:.3f}, model={n}")
            logger.info("\nStart Evolution...")

        logger.info(f"\nSearching with {self.constraint_type} constraint ({self.efficiency_constraint:.3f})")
        for it in range(max_time_budget):
        # for it in tqdm(range(max_time_budget),
        #                desc=f"Searching with {self.constraint_type} constraint ({self.efficiency_constraint:.3f})"):
            parents = sorted(population, key=lambda x: finite_or_default(x[0], -np.inf))[::-1][:parents_size]
            acc_top = finite_or_default(parents[0][0], 0.0)
            if verbose:
                logger.info("Iter: %d Acc: %.6f", it + 1, acc_top)

            if acc_top > best_valids[-1]:
                best_valids.append(acc_top)
                best_info = parents[0]
            else:
                best_valids.append(best_valids[-1])

            population = parents[:]
            child_pool = []
            efficiency_pool = []

            for _ in range(mutation_numbers):
                par_sample = parents[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(clamp_min(efficiency))

            for _ in range(population_size - mutation_numbers):
                par_sample1 = parents[np.random.randint(parents_size)][1]
                par_sample2 = parents[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(clamp_min(efficiency))

            logger.info(f"[{it+1}|{max_time_budget}] Mutate and Crossover...")
            for i in range(population_size):
            # for i in trange(population_size, desc=f"[{it+1}|{max_time_budget}] Mutate and Crossover..."):
                logger.info(f"----{i}/{population_size-1}")
                subnet, acc = self.predict_accuracy_safe(child_pool[i])
                acc = finite_or_default(acc, default=0.0)
                population.append((acc, child_pool[i], efficiency_pool[i], subnet))

        return best_valids, best_info  # best_acc_history, (acc, config, eff, subnet)