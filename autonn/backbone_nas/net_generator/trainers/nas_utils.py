'''
NAS utils
'''

from collections import defaultdict

import numpy as np

VIS_DICT = {}


def legal(cand, states):
    '''
    legal
    '''
    assert isinstance(cand, tuple) and len(cand) == len(states)
    if cand not in VIS_DICT:
        VIS_DICT[cand] = {}
    info = VIS_DICT[cand]
    if 'visited' in info:
        return False

    VIS_DICT[cand] = info

    return True


def stack_random_cand(random_func, *, batchsize=10):
    '''
    stack random cand
    '''
    while True:
        cands = [random_func() for _ in range(batchsize)]
        for cand in cands:
            if cand not in VIS_DICT:
                VIS_DICT[cand] = {}
            # info = vis_dict[cand]

        for cand in cands:
            yield cand


def random_can(num, states):   # num=50
    '''
    rand candidate
    '''
    print('random select ........')
    candidates = []
    cand_iter = stack_random_cand(lambda: tuple(
        np.random.randint(i) for i in states))
    while len(candidates) < num:
        cand = next(cand_iter)

        if not legal(cand, states):
            continue
        #########################################################
        # expansion rate, stride predefined
        # cand
        #########################################################
        candidates.append(cand)
        # print('random {}/{}'.format(len(candidates),num))

    # print('random_num = {}'.format(len(candidates)))
    return candidates


def crowding_dist(fitness=None):
    """
    :param fitness: A list of fitness values
    :return: A list of crowding distances of chrmosomes

    The crowding-distance computation requires sorting the population
    according to each objective function value
    in ascending order of magnitude.
    Thereafter, for each objective function,
    the boundary solutions
    (solutions with smallest and largest function values)
    are assigned an infinite distance value.
    All other intermediate solutions are assigned a distance value equal to
    the absolute normalized difference
    in the function values of two adjacent solutions.
    """

    # initialize list: [0.0, 0.0, 0.0, ...]
    distances = [0.0] * len(fitness)
    crowd = [(f_value, i) for i, f_value in enumerate(
        fitness)]  # create keys for fitness values

    n_obj = len(fitness[0])

    for _i in range(n_obj):  # calculate for each objective
        crowd.sort(key=lambda element, i=_i: element[0][i])
        # After sorting,  boundary solutions are assigned Inf
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        distances[crowd[0][1]] = float("Inf")
        distances[crowd[-1][1]] = float("inf")
        # If objective values are same, skip this loop
        if crowd[-1][0][_i] == crowd[0][0][_i]:
            continue
        # normalization (max - min) as Denominator
        norm = float(crowd[-1][0][_i] - crowd[0][0][_i])
        # crowd: [([obj_1, obj_2, ...], i_0), ([obj_1, obj_2, ...], i_1), ...]
        # calculate each individual's Crowding Distance of i th objective
        # technique: shift the list and zip
        for prev, cur, _next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            # sum up the distance of ith individual
            # along each of the objectives
            distances[cur[1]] += (_next[0][_i] - prev[0][_i]) / norm

    return distances


def dominates(obj1, obj2, sign=None):
    """Return true if each objective of *self* is not strictly worse than
            the corresponding objective of *other* and
            at least one objective is strictly better.
        **no need to care about the equal cases
        (Cuz equal cases mean they are non-dominators)
    :param obj1: a list of multiple objective values
    :type obj1: numpy.ndarray
    :param obj2: a list of multiple objective values
    :type obj2: numpy.ndarray
    :param sign: target types. positive means maximize and otherwise minimize.
    :type sign: list
    """
    if sign is None:
        sign = [-1, -1]
    indicator = False
    for _a, _b, _sign in zip(obj1, obj2, sign):
        if _a * _sign > _b * _sign:
            indicator = True
        # if one of the objectives is dominated, then return False
        elif _a * _sign < _b * _sign:
            return False
    return indicator


def sort_nondominated(fitness, k=None, first_front_only=False):
    """Sort the first *k* *individuals* into different nondomination levels
        using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
        see [Deb2002]_.
        This algorithm has a time complexity of :math:`O(MN^2)`,
        where :math:`M` is the number of objectives and :math:`N` the number of
        individuals.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param first_front_only: If :obj:`True` sort only the first front and
                                    exit.
        :param sign: indicate the objectives are maximized or minimized
        :returns: A list of Pareto fronts (lists), the first list includes
                    nondominated individuals.
        .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
            non-dominated sorting genetic algorithm for multi-objective
            optimization: NSGA-II", 2002.
    """
    if k is None:
        k = len(fitness)

    # Use objectives as keys to make python dictionary
    map_fit_ind = defaultdict(list)
    # fitness = [(1, 2), (2, 2), (3, 1), (1, 4), (1, 1)...]
    for i, f_value in enumerate(fitness):
        map_fit_ind[f_value].append(i)
    fits = list(map_fit_ind.keys())  # fitness values

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)  # n (The number of people dominate you)
    dominated_fits = defaultdict(list)  # Sp (The people you dominate)

    # Rank first Pareto front
    # *fits* is a iterable list of chromosomes. Each has multiple objectives.
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i + 1:]:
            # Eventhougn equals or empty list, n & Sp won't be affected
            if dominates(fit_i, fit_j):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif dominates(fit_j, fit_i):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]  # The first front
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all individuals are sorted or
    # the given number of individual are sorted.
    # If Sn=0 then the set of objectives belongs to the next front
    if not first_front_only:  # first front only
        _n = min(len(fitness), k)
        while pareto_sorted < _n:
            fronts.append([])
            for fit_p in current_front:
                # Iterate Sn in current fronts
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1  # Next front -> Sn - 1
                    if dominating_fits[fit_d] == 0:  # Sn=0 -> next front
                        next_front.append(fit_d)
                        # Count and append chromosomes with same objectives
                        pareto_sorted += len(map_fit_ind[fit_d])
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts
