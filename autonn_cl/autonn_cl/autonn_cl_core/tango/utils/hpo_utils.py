from __future__ import annotations

import math
import numpy as np

from types import SimpleNamespace
from typing import Any, Dict, Optional, NamedTuple, Tuple, cast
from typing_extensions import Literal, TypedDict, NotRequired
from enum import Enum

import json_tricks

import logging
_logger = logging.getLogger(__name__)

# check duplicate --------------------------------------------------------------
class Deduplicator:
    """
    A helper for tuners to deduplicate generated parameters.

    When the tuner generates an already existing parameter,
    calling this will return a new parameter generated with grid search.
    Otherwise it returns the orignial parameter object.

    If all parameters have been generated, raise ``NoMoreTrialError``.

    All search space types, including nested choice, are supported.

    Resuming and updating search space are not supported for now.
    It will not raise error, but may return duplicate parameters.

    See random tuner's source code for example usage.
    """

    def __init__(self, formatted_search_space: FormattedSearchSpace):
        self._space: FormattedSearchSpace = formatted_search_space
        self._never_dup: bool = any(_spec_never_dup(spec) for spec in self._space.values())
        self._history: set[str] = set()
        self._grid_search: GridSearchTuner | None = None

    def __call__(self, formatted_parameters: FormattedParameters) -> FormattedParameters:
        if self._never_dup or self._not_dup(formatted_parameters):
            return formatted_parameters

        if self._grid_search is None:
            _logger.info(f'Tuning algorithm generated duplicate parameter: {formatted_parameters}')
            _logger.info(f'Use grid search for deduplication.')
            self._init_grid_search()

        while True:
            new = self._grid_search._suggest()  # type: ignore
            if new is None:
                raise nni.NoMoreTrialError()
            if self._not_dup(new):
                return new

    def _init_grid_search(self) -> None:
        from nni.algorithms.hpo.gridsearch_tuner import GridSearchTuner
        self._grid_search = GridSearchTuner()
        self._grid_search.history = self._history
        self._grid_search.space = self._space
        self._grid_search._init_grid()

    def _not_dup(self, formatted_parameters: FormattedParameters) -> bool:
        params = deformat_parameters(formatted_parameters, self._space)
        params_str = typing.cast(str, nni.dump(params, sort_keys=True))
        if params_str in self._history:
            return False
        else:
            self._history.add(params_str)
            return True

    def add_history(self, formatted_parameters: FormattedParameters) -> None:
        params = deformat_parameters(formatted_parameters, self._space)
        params_str = typing.cast(str, nni.dump(params, sort_keys=True))
        if params_str not in self._history:
            self._history.add(params_str)

def _spec_never_dup(spec: ParameterSpec) -> bool:
    if spec.is_nested():
        return False  # "not chosen" duplicates with "not chosen"
    if spec.categorical or spec.q is not None:
        return False
    if spec.normal_distributed:
        return spec.sigma > 0
    else:
        return spec.low < spec.high

# check search space -----------------------------------------------------------
common_search_space_types = [
    'choice',
    'randint',
    'uniform',
    'quniform',
    'loguniform',
    'qloguniform',
    'normal',
    'qnormal',
    'lognormal',
    'qlognormal',
]

def validate_search_space(
        search_space: Any,
        support_types: list[str] | None = None,
        raise_exception: bool = False  # for now, in case false positive
    ) -> bool:

    if not raise_exception:
        try:
            validate_search_space(search_space, support_types, True)
            return True
        except ValueError as e:
            logging.getLogger(__name__).error(e.args[0])
        return False

    if support_types is None:
        support_types = common_search_space_types

    if not isinstance(search_space, dict):
        raise ValueError(f'search space is a {type(search_space).__name__}, expect a dict : {repr(search_space)}')

    for name, spec in search_space.items():
        if not isinstance(spec, dict):
            raise ValueError(f'search space "{name}" is a {type(spec).__name__}, expect a dict : {repr(spec)}')
        if '_type' not in spec or '_value' not in spec:
            raise ValueError(f'search space "{name}" does not have "_type" or "_value" : {spec}')
        type_ = spec['_type']
        if type_ not in support_types:
            raise ValueError(f'search space "{name}" has unsupported type "{type_}" : {spec}')
        args = spec['_value']
        if not isinstance(args, list):
            raise ValueError(f'search space "{name}"\'s value is not a list : {spec}')

        if type_ == 'choice':
            if not all(isinstance(arg, (float, int, str)) for arg in args):
                # FIXME: need further check for each algorithm which types are actually supported
                # for now validation only prints warning so it doesn't harm
                if not isinstance(args[0], dict) or '_name' not in args[0]:  # not nested search space
                    raise ValueError(f'search space "{name}" (choice) should only contain numbers or strings : {spec}')
            continue

        if type_.startswith('q'):
            if len(args) != 3:
                raise ValueError(f'search space "{name}" ({type_}) must have 3 values : {spec}')
        else:
            if len(args) != 2:
                raise ValueError(f'search space "{name}" ({type_}) must have 2 values : {spec}')

        if type_ == 'randint':
            if not all(isinstance(arg, int) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have int values : {spec}')
        else:
            if not all(isinstance(arg, (float, int)) for arg in args):
                raise ValueError(f'search space "{name}" ({type_}) must have float values : {spec}')

        if 'normal' not in type_:
            if args[0] >= args[1]:
                raise ValueError(f'search space "{name}" ({type_}) must have high > low : {spec}')
            if 'log' in type_ and args[0] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have low > 0 : {spec}')
        else:
            if args[1] <= 0:
                raise ValueError(f'search space "{name}" ({type_}) must have sigma > 0 : {spec}')

    return True


# parameter spec ---------------------------------------------------------------
ParameterKey = Tuple['str | int', ...]
FormattedParameters = Dict[ParameterKey, 'float | int']
FormattedSearchSpace = Dict[ParameterKey, 'ParameterSpec']

class ParameterSpec(NamedTuple):
    """
    Specification (aka space / range / domain) of one single parameter.

    NOTE: For `loguniform` (and `qloguniform`), the fields `low` and `high` are logarithm of original values.
    """

    name: str                       # The object key in JSON
    type: str                       # "_type" in JSON
    values: list[Any]               # "_value" in JSON

    key: ParameterKey               # The "path" of this parameter

    categorical: bool               # Whether this paramter is categorical (unordered) or numerical (ordered)
    size: int = cast(int, None)     # If it's categorical, how many candidates it has
    chosen_size: int | None = 1     # If it's categorical, it should choose how many candidates.
                                    # By default, 1. If none, arbitrary number of candidates can be chosen.

    # uniform distributed
    low: float = cast(float, None)  # Lower bound of uniform parameter
    high: float = cast(float, None) # Upper bound of uniform parameter

    normal_distributed: bool = cast(bool, None)
                                    # Whether this parameter is uniform or normal distrubuted
    mu: float = cast(float, None)   # µ of normal parameter
    sigma: float = cast(float, None)# σ of normal parameter

    q: float | None = None          # If not `None`, the parameter value should be an integer multiple of this
    clip: tuple[float, float] | None = None
                                    # For q(log)uniform, this equals to "values[:2]"; for others this is None

    log_distributed: bool = cast(bool, None)
                                    # Whether this parameter is log distributed
                                    # When true, low/high/mu/sigma describes log of parameter value (like np.lognormal)

    def is_activated_in(self, partial_parameters: FormattedParameters) -> bool:
        """
        For nested search space, check whether this parameter should be skipped for current set of paremters.
        This function must be used in a pattern similar to random tuner. Otherwise it will misbehave.
        """
        if self.is_nested():
            return partial_parameters[self.key[:-2]] == self.key[-2]
        else:
            return True

    def is_nested(self):
        """
        Check whether this parameter is inside a nested choice.
        """
        return len(self.key) >= 2 and isinstance(self.key[-2], int)

def format_search_space(search_space: SearchSpace) -> FormattedSearchSpace:
    """
    Convert user provided search space into a dict of ParameterSpec.
    The dict key is dict value's `ParameterSpec.key`.
    """
    formatted = _format_search_space(tuple(), search_space)
    return {spec.key: spec for spec in formatted}

def deformat_parameters(
        formatted_parameters: FormattedParameters,
        formatted_search_space: FormattedSearchSpace) -> Parameters:
    """
    Convert internal format parameters to users' expected format.

    "test/ut/sdk/test_hpo_formatting.py" provides examples of how this works.

    The function do following jobs:
     1. For "choice" and "randint", convert index (integer) to corresponding value.
     2. For "*log*", convert x to `exp(x)`.
     3. For "q*", convert x to `round(x / q) * q`, then clip into range.
     4. For nested choices, convert flatten key-value pairs into nested structure.
    """
    ret: Parameters = {}
    for key, x in formatted_parameters.items():
        spec = formatted_search_space[key]
        if spec.categorical:
            x = cast(int, x)
            if spec.type == 'randint':
                lower = min(math.ceil(float(x)) for x in spec.values)
                _assign(ret, key, int(lower + x))
            elif _is_nested_choices(spec.values):
                _assign(ret, tuple([*key, '_name']), spec.values[x]['_name'])
            else:
                _assign(ret, key, spec.values[x])
        else:
            if spec.log_distributed:
                x = math.exp(x)
            if spec.q is not None:
                x = round(x / spec.q) * spec.q
            if spec.clip:
                x = max(x, spec.clip[0])
                x = min(x, spec.clip[1])
            if isinstance(x, np.number):
                x = x.item()
            _assign(ret, key, x)
    return ret

def format_parameters(parameters: Parameters, formatted_search_space: FormattedSearchSpace) -> FormattedParameters:
    """
    Convert end users' parameter format back to internal format, mainly for resuming experiments.

    The result is not accurate for "q*" and for "choice" that have duplicate candidates.
    """
    # I don't like this function. It's better to use checkpoint for resuming.
    ret = {}
    for key, spec in formatted_search_space.items():
        if not spec.is_activated_in(ret):
            continue
        value: Any = parameters
        for name in key:
            if isinstance(name, str):
                value = value[name]
        if spec.categorical:
            if spec.type == 'randint':
                lower = min(math.ceil(float(x)) for x in spec.values)
                ret[key] = value - lower
            elif _is_nested_choices(spec.values):
                names = [nested['_name'] for nested in spec.values]
                ret[key] = names.index(value['_name'])
            else:
                ret[key] = spec.values.index(value)
        else:
            if spec.log_distributed:
                value = math.log(value)
            ret[key] = value
    return ret

def _format_search_space(parent_key: ParameterKey, space: SearchSpace) -> list[ParameterSpec]:
    formatted: list[ParameterSpec] = []
    for name, spec in space.items():
        if name == '_name':
            continue
        key = tuple([*parent_key, name])
        formatted.append(_format_parameter(key, spec['_type'], spec['_value']))
        if spec['_type'] == 'choice' and _is_nested_choices(spec['_value']):
            for index, sub_space in enumerate(spec['_value']):
                key = tuple([*parent_key, name, index])
                formatted += _format_search_space(key, sub_space)
    return formatted

def _format_parameter(key: ParameterKey, type_: str, values: list[Any]):
    spec = SimpleNamespace(
        name = key[-1],
        type = type_,
        values = values,
        key = key,
        categorical = type_ in ['choice', 'randint'],
    )

    if spec.categorical:
        if type_ == 'choice':
            spec.size = len(values)
        else:
            lower = math.ceil(float(values[0]))
            upper = math.ceil(float(values[1]))
            spec.size = upper - lower

    else:
        if type_.startswith('q'):
            spec.q = float(values[2])
        else:
            spec.q = None
        spec.log_distributed = ('log' in type_)

        if 'normal' in type_:
            spec.normal_distributed = True
            spec.mu = float(values[0])
            spec.sigma = float(values[1])

        else:
            spec.normal_distributed = False
            spec.low = float(values[0])
            spec.high = float(values[1])
            if spec.q is not None:
                spec.clip = (spec.low, spec.high)
            if spec.log_distributed:
                # make it align with mu
                spec.low = math.log(spec.low)
                spec.high = math.log(spec.high)

    return ParameterSpec(**spec.__dict__)

def _is_nested_choices(values: list[Any]) -> bool:
    assert values  # choices should not be empty
    for value in values:
        if not isinstance(value, dict):
            return False
        if '_name' not in value:
            return False
    return True

def _assign(params: Parameters, key: ParameterKey, x: Any) -> None:
    if len(key) == 1:
        params[cast(str, key[0])] = x
    elif isinstance(key[0], int):
        _assign(params, key[1:], x)
    else:
        if key[0] not in params:
            params[key[0]] = {}
        _assign(params[key[0]], key[1:], x)

class OptimizeMode(Enum):
    Minimize = 'minimize'
    Maximize = 'maximize'


# tuner ------------------------------------------------------------------------
class Recoverable:
    def __init__(self):
        self.recovered_max_param_id = -1
        self.recovered_trial_params = {}

    def load_checkpoint(self) -> None:
        pass

    def save_checkpoint(self) -> None:
        pass

    def get_checkpoint_path(self) -> str | None:
        ckp_path = os.getenv('NNI_CHECKPOINT_DIRECTORY')
        if ckp_path is not None and os.path.isdir(ckp_path):
            return ckp_path
        return None

    def recover_parameter_id(self, data) -> int:
        # this is for handling the resuming of the interrupted data: parameters
        if not isinstance(data, list):
            data = [data]

        previous_max_param_id = 0
        for trial in data:
            # {'parameter_id': 0, 'parameter_source': 'resumed', 'parameters': {'batch_size': 128, ...}
            if isinstance(trial, str):
                trial = nni.load(trial)
            if not isinstance(trial['parameter_id'], int):
                # for dealing with user customized trials
                # skip for now
                continue
            self.recovered_trial_params[trial['parameter_id']] = trial['parameters']
            if previous_max_param_id < trial['parameter_id']:
                previous_max_param_id = trial['parameter_id']
        self.recovered_max_param_id = previous_max_param_id
        return previous_max_param_id

    def is_created_in_previous_exp(self, param_id: int | None) -> bool:
        if param_id is None:
            return False
        return param_id <= self.recovered_max_param_id

    def get_previous_param(self, param_id: int) -> dict:
        return self.recovered_trial_params[param_id]

class Tuner(Recoverable):
    """
    Tuner is an AutoML algorithm, which generates a new configuration for the next try.
    A new trial will run with this configuration.

    This is the abstract base class for all tuners.
    Tuning algorithms should inherit this class and override :meth:`update_search_space`, :meth:`receive_trial_result`,
    as well as :meth:`generate_parameters` or :meth:`generate_multiple_parameters`.

    After initializing, NNI will first call :meth:`update_search_space` to tell tuner the feasible region,
    and then call :meth:`generate_parameters` one or more times to request for hyper-parameter configurations.

    The framework will train several models with given configuration.
    When one of them is finished, the final accuracy will be reported to :meth:`receive_trial_result`.
    And then another configuration will be reqeusted and trained, util the whole experiment finish.

    If a tuner want's to know when a trial ends, it can also override :meth:`trial_end`.

    Tuners use *parameter ID* to track trials.
    In tuner context, there is a one-to-one mapping between parameter ID and trial.
    When the framework ask tuner to generate hyper-parameters for a new trial,
    an ID has already been assigned and can be recorded in :meth:`generate_parameters`.
    Later when the trial ends, the ID will be reported to :meth:`trial_end`,
    and :meth:`receive_trial_result` if it has a final result.
    Parameter IDs are unique integers.

    The type/format of search space and hyper-parameters are not limited,
    as long as they are JSON-serializable and in sync with trial code.
    For HPO tuners, however, there is a widely shared common interface,
    which supports ``choice``, ``randint``, ``uniform``, and so on.
    See ``docs/en_US/Tutorial/SearchSpaceSpec.md`` for details of this interface.

    [WIP] For advanced tuners which take advantage of trials' intermediate results,
    an ``Advisor`` interface is under development.

    See Also
    --------
    Builtin tuners:
    :class:`~nni.algorithms.hpo.hyperopt_tuner.hyperopt_tuner.HyperoptTuner`
    :class:`~nni.algorithms.hpo.evolution_tuner.evolution_tuner.EvolutionTuner`
    :class:`~nni.algorithms.hpo.smac_tuner.SMACTuner`
    :class:`~nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner`
    :class:`~nni.algorithms.hpo.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner`
    :class:`~nni.algorithms.hpo.metis_tuner.mets_tuner.MetisTuner`
    :class:`~nni.algorithms.hpo.ppo_tuner.PPOTuner`
    :class:`~nni.algorithms.hpo.gp_tuner.gp_tuner.GPTuner`
    """

    def generate_parameters(self, parameter_id: int, **kwargs) -> Parameters:
        """
        Abstract method which provides a set of hyper-parameters.

        This method will get called when the framework is about to launch a new trial,
        if user does not override :meth:`generate_multiple_parameters`.

        The return value of this method will be received by trials via :func:`nni.get_next_parameter`.
        It should fit in the search space, though the framework will not verify this.

        User code must override either this method or :meth:`generate_multiple_parameters`.

        Parameters
        ----------
        parameter_id : int
            Unique identifier for requested hyper-parameters. This will later be used in :meth:`receive_trial_result`.
        **kwargs
            Unstable parameters which should be ignored by normal users.

        Returns
        -------
        any
            The hyper-parameters, a dict in most cases, but could be any JSON-serializable type when needed.

        Raises
        ------
        nni.NoMoreTrialError
            If the search space is fully explored, tuner can raise this exception.
        """
        # FIXME: some tuners raise NoMoreTrialError when they are waiting for more trial results
        # we need to design a new exception for this purpose
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def generate_multiple_parameters(self, parameter_id_list: list[int], **kwargs) -> list[Parameters]:
        """
        Callback method which provides multiple sets of hyper-parameters.

        This method will get called when the framework is about to launch one or more new trials.

        If user does not override this method, it will invoke :meth:`generate_parameters` on each parameter ID.

        See :meth:`generate_parameters` for details.

        User code must override either this method or :meth:`generate_parameters`.

        Parameters
        ----------
        parameter_id_list : list of int
            Unique identifiers for each set of requested hyper-parameters.
            These will later be used in :meth:`receive_trial_result`.
        **kwargs
            Unstable parameters which should be ignored by normal users.

        Returns
        -------
        list
            List of hyper-parameters. An empty list indicates there are no more trials.
        """
        result = []
        for parameter_id in parameter_id_list:
            try:
                _logger.debug("generating param for %s", parameter_id)
                res = self.generate_parameters(parameter_id, **kwargs)
            except nni.NoMoreTrialError:
                return result
            result.append(res)
        return result

    def receive_trial_result(self, parameter_id: int, parameters: Parameters, value: TrialMetric, **kwargs) -> None:
        """
        Abstract method invoked when a trial reports its final result. Must override.

        This method only listens to results of algorithm-generated hyper-parameters.
        Currently customized trials added from web UI will not report result to this method.

        Parameters
        ----------
        parameter_id : int
            Unique identifier of used hyper-parameters, same with :meth:`generate_parameters`.
        parameters
            Hyper-parameters generated by :meth:`generate_parameters`.
        value
            Result from trial (the return value of :func:`nni.report_final_result`).
        **kwargs
            Unstable parameters which should be ignored by normal users.
        """
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def _accept_customized_trials(self, accept=True):
        # FIXME: because Tuner is designed as interface, this API should not be here

        # Enable or disable receiving results of user-added hyper-parameters.
        # By default `receive_trial_result()` will only receive results of algorithm-generated hyper-parameters.
        # If tuners want to receive those of customized parameters as well, they can call this function in `__init__()`.

        # pylint: disable=attribute-defined-outside-init
        self._accept_customized = accept

    def trial_end(self, parameter_id: int, success: bool, **kwargs) -> None:
        """
        Abstract method invoked when a trial is completed or terminated. Do nothing by default.

        Parameters
        ----------
        parameter_id : int
            Unique identifier for hyper-parameters used by this trial.
        success : bool
            True if the trial successfully completed; False if failed or terminated.
        **kwargs
            Unstable parameters which should be ignored by normal users.
        """

    def update_search_space(self, search_space: SearchSpace) -> None:
        """
        Abstract method for updating the search space. Must override.

        Tuners are advised to support updating search space at run-time.
        If a tuner can only set search space once before generating first hyper-parameters,
        it should explicitly document this behaviour.

        Parameters
        ----------
        search_space
            JSON object defined by experiment owner.
        """
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def load_checkpoint(self) -> None:
        """
        Internal API under revising, not recommended for end users.
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Load checkpoint ignored by tuner, checkpoint path: %s', checkpoin_path)

    def save_checkpoint(self) -> None:
        """
        Internal API under revising, not recommended for end users.
        """
        checkpoin_path = self.get_checkpoint_path()
        _logger.info('Save checkpoint ignored by tuner, checkpoint path: %s', checkpoin_path)

    def import_data(self, data: list[TrialRecord]) -> None:
        """
        Internal API under revising, not recommended for end users.
        """
        # Import additional data for tuning
        # data: a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'
        pass

    def _on_exit(self) -> None:
        pass

    def _on_error(self) -> None:
        pass

class GridSearchTuner(Tuner):
    """
    Grid search tuner divides search space into evenly spaced grid, and performs brute-force traverse.

    Recommended when the search space is small, or if you want to find strictly optimal hyperparameters.

    **Implementation**

    The original grid search approach performs an exhaustive search through a space consists of ``choice`` and ``randint``.

    NNI's implementation extends grid search to support all search spaces types.

    When the search space contains continuous parameters like ``normal`` and ``loguniform``,
    grid search tuner works in following steps:

    1. Divide the search space into a grid.
    2. Perform an exhaustive searth through the grid.
    3. Subdivide the grid into a finer-grained new grid.
    4. Goto step 2, until experiment end.

    As a deterministic algorithm, grid search has no argument.

    Examples
    --------

    .. code-block::

        config.tuner.name = 'GridSearch'
    """

    def __init__(self, optimize_mode=None):
        self.space = None

        # the grid to search in this epoch
        # when the space is fully explored, grid is set to None
        self.grid = None  # list[int | float]

        # a paremter set is internally expressed as a vector
        # for each dimension i, self.vector[i] is the parameter's index in self.grid[i]
        # in second epoch of above example, vector [1, 2, 0] means parameters {x: 7, y: 0.67, z: 2}
        self.vector = None  # list[int]

        # this tells which parameters are derived from previous epoch
        # in second epoch of above example, epoch_bar is [2, 1, 1]
        self.epoch_bar = None  # list[int]

        # this stores which intervals are possibly divisible (low < high after log and q)
        # in first epoch of above example, divisions are:
        #     {1: [(0,1/2), (1/2,1)], 2: [(1/2,1)]}
        # in second epoch:
        #     {1: [(0,1/4), (1/4,1/2), (1/2,3/4), (3/4,1)], 2: [(1/2,3/4)]}
        # and in third epoch:
        #     {1: [(0,1/8), ..., (7/8,1)], 2: []}
        self.divisions = {}  # dict[int, list[tuple[float, float]]]

        # dumped JSON string of all tried parameters
        self.history = set()

        if optimize_mode is not None:
            _logger.info(f'Ignored optimize_mode "{optimize_mode}"')

    def update_search_space(self, space):
        self.space = format_search_space(space)
        if not self.space:  # the tuner will crash in this case, report it explicitly
            raise ValueError('Search space is empty')
        self._init_grid()

    def generate_parameters(self, *args, **kwargs):
        while True:
            params = self._suggest()
            if params is None:
                raise nni.NoMoreTrialError('Search space fully explored')
            params = deformat_parameters(params, self.space)

            params_str = nni.dump(params, sort_keys=True)
            if params_str not in self.history:
                self.history.add(params_str)
                return params

    def receive_trial_result(self, *args, **kwargs):
        pass

    def import_data(self, data):
        # TODO
        # use tuple to dedup in case of order/precision issue causes matching failed
        # and remove `epoch_bar` to use uniform dedup mechanism
        for trial in data:
            params_str = nni.dump(trial['parameter'], sort_keys=True)
            self.history.add(params_str)

    def _suggest(self):
        # returns next parameter set, or None if the space is already fully explored
        while True:
            if self.grid is None:  # search space fully explored
                return None

            self._next_vector()

            if self.vector is None:  # epoch end, update grid and retry
                self._next_grid()
                continue

            old = all((self.vector[i] < self.epoch_bar[i]) for i in range(len(self.space)))
            if old:  # already explored in past epochs
                continue

            # this vector is valid, stop
            _logger.debug(f'vector: {self.vector}')
            return self._current_parameters()

    def _next_vector(self):
        # iterate to next vector of this epoch, set vector to None if epoch end
        if self.vector is None:  # first vector in this epoch
            self.vector = [0] * len(self.space)
            return

        # deal with nested choice, don't touch nested spaces that are not chosen by current vector
        activated_dims = []
        params = self._current_parameters()
        for i, spec in enumerate(self.space.values()):
            if spec.is_activated_in(params):
                activated_dims.append(i)

        for i in reversed(activated_dims):
            if self.vector[i] + 1 < len(self.grid[i]):
                self.vector[i] += 1
                return
            else:
                self.vector[i] = 0

        self.vector = None  # the loop ends without returning, no more vector in this epoch

    def _next_grid(self):
        # update grid information (grid, epoch_bar, divisions) for next epoch
        updated = False
        for i, spec in enumerate(self.space.values()):
            self.epoch_bar[i] = len(self.grid[i])
            if not spec.categorical:
                # further divide intervals
                new_vals = []  # values to append to grid
                new_divs = []  # sub-intervals
                for l, r in self.divisions[i]:
                    mid = (l + r) / 2
                    diff_l = _less(l, mid, spec)
                    diff_r = _less(mid, r, spec)
                    # if l != 0 and r != 1, then they are already in the grid, else they are not
                    # the special case is needed because for normal distribution 0 and 1 will generate infinity
                    if (diff_l or l == 0.0) and (diff_r or r == 1.0):
                        # we can skip these for non-q, but it will complicate the code
                        new_vals.append(mid)
                        updated = True
                    if diff_l:
                        new_divs.append((l, mid))
                        updated = (updated or l == 0.0)
                    if diff_r:
                        new_divs.append((mid, r))
                        updated = (updated or r == 1.0)
                self.grid[i] += new_vals
                self.divisions[i] = new_divs

        if not updated:  # fully explored
            _logger.info('Search space has been fully explored')
            self.grid = None
        else:
            size = _grid_size_info(self.grid)
            _logger.info(f'Grid subdivided, new size: {size}')

    def _init_grid(self):
        self.epoch_bar = [0 for _ in self.space]
        self.grid = [None for _ in self.space]
        for i, spec in enumerate(self.space.values()):
            if spec.categorical:
                self.grid[i] = list(range(spec.size))
            else:
                self.grid[i] = [0.5]
                self.divisions[i] = []
                if _less(0, 0.5, spec):
                    self.divisions[i].append((0, 0.5))
                if _less(0.5, 1, spec):
                    self.divisions[i].append((0.5, 1))

        size = _grid_size_info(self.grid)
        _logger.info(f'Grid initialized, size: {size}')

    def _current_parameters(self):
        # convert self.vector to "formatted" parameters
        params = {}
        for i, spec in enumerate(self.space.values()):
            if spec.is_activated_in(params):
                x = self.grid[i][self.vector[i]]
                if spec.categorical:
                    params[spec.key] = x
                else:
                    params[spec.key] = _cdf_inverse(x, spec)
        return params

def _less(x, y, spec):
    #if spec.q is None:  # TODO: comment out because of edge case UT uniform(99.9, 99.9)
    #    return x < y
    real_x = _deformat_single_parameter(_cdf_inverse(x, spec), spec)
    real_y = _deformat_single_parameter(_cdf_inverse(y, spec), spec)
    return real_x < real_y

def _cdf_inverse(x, spec):
    # inverse function of spec's cumulative distribution function
    if spec.normal_distributed:
        return spec.mu + spec.sigma * math.sqrt(2) * erfinv(2 * x - 1)
    else:
        return spec.low + (spec.high - spec.low) * x

def _deformat_single_parameter(x, spec):
    if math.isinf(x):
        return x
    spec_dict = spec._asdict()
    spec_dict['key'] = (spec.name,)
    spec = ParameterSpec(**spec_dict)
    params = deformat_parameters({spec.key: x}, {spec.key: spec})
    return params[spec.name]

def _grid_size_info(grid):
    if len(grid) == 1:
        return str(len(grid[0]))
    sizes = [len(candidates) for candidates in grid]
    mul = '×'.join(str(s) for s in sizes)
    total = np.prod(sizes)
    return f'({mul}) = {total}'


# parameters -------------------------------------------------------------------
Parameters = Dict[str, Any]

class _ParameterSearchSpace(TypedDict):
    _type: Literal[
        'choice', 'randint',
        'uniform', 'loguniform', 'quniform', 'qloguniform',
        'normal', 'lognormal', 'qnormal', 'qlognormal',
    ]
    _value: List[Any]

SearchSpace = Dict[str, _ParameterSearchSpace]
TrialMetric = float

class TrialRecord(TypedDict):
    parameter: Parameters
    value: TrialMetric

class ParameterRecord(TypedDict):
    """The format which is used to record parameters at NNI manager side.

    :class:`~nni.runtime.msg_dispatcher.MsgDispatcher` packs the parameters generated by tuners
    into a :class:`ParameterRecord` and sends it to NNI manager.
    NNI manager saves the tuner into database and sends it to trial jobs when they ask for parameters.
    :class:`~nni.runtime.trial_command_channel.TrialCommandChannel` receives the :class:`ParameterRecord`
    and then hand it over to trial.

    Most users don't need to use this class directly.
    """
    parameter_id: Optional[int]
    parameters: Parameters
    parameter_source: NotRequired[Literal['algorithm', 'customized', 'resumed']]

    # NOTE: in some cases the record might contain extra fields,
    # but they are undocumented and should not be used by users.
    parameter_index: NotRequired[int]
    trial_job_id: NotRequired[str]
    version_info: NotRequired[dict]


# utils ------------------------------------------------------------------------
def dump(obj: Any, fp: Optional[Any] = None, *, use_trace: bool = True, pickle_size_limit: int = 4096,
         allow_nan: bool = True, **json_tricks_kwargs) -> str:
    """
    Convert a nested data structure to a json string. Save to file if fp is specified.
    Use json-tricks as main backend. For unhandled cases in json-tricks, use cloudpickle.
    The serializer is not designed for long-term storage use, but rather to copy data between processes.
    The format is also subject to change between NNI releases.

    It's recommended to use ``dump`` with ``trace``. The traced object can be stored with their traced arguments.
    For more complex objects, it will look for ``_dump`` and ``_load`` pair in the class.
    If not found, it will fallback to binary dump with cloudpickle.

    To compress the payload, please use :func:`dump_bytes`.

    Parameters
    ----------
    obj : any
        The object to dump.
    fp : file handler or path
        File to write to. Keep it none if you want to dump a string.
    pickle_size_limit : int
        This is set to avoid too long serialization result. Set to -1 to disable size check.
    allow_nan : bool
        Whether to allow nan to be serialized. Different from default value in json-tricks, our default value is true.
    json_tricks_kwargs : dict
        Other keyword arguments passed to json tricks (backend), e.g., indent=2.

    Returns
    -------
    str or bytes
        Normally str. Sometimes bytes (if compressed).
    """

    if json_tricks_kwargs.get('compression') is not None:
        raise ValueError('If you meant to compress the dumped payload, please use `dump_bytes`.')
    result = _dump(
        obj=obj,
        fp=fp,
        use_trace=use_trace,
        pickle_size_limit=pickle_size_limit,
        allow_nan=allow_nan,
        **json_tricks_kwargs)
    return cast(str, result)


def dump_bytes(obj: Any, fp: Optional[Any] = None, *, compression: int = cast(int, None),
               use_trace: bool = True, pickle_size_limit: int = 4096,
               allow_nan: bool = True, **json_tricks_kwargs) -> bytes:
    """
    Same as :func:`dump`, but to comporess payload, with `compression <https://json-tricks.readthedocs.io/en/stable/#dump>`__.
    """
    if compression is None:
        raise ValueError('compression must be set.')
    result = _dump(
        obj=obj,
        fp=fp,
        compression=compression,
        use_trace=use_trace,
        pickle_size_limit=pickle_size_limit,
        allow_nan=allow_nan,
        **json_tricks_kwargs)
    return cast(bytes, result)


def _dump(*, obj: Any, fp: Optional[Any], use_trace: bool, pickle_size_limit: int,
          allow_nan: bool, **json_tricks_kwargs) -> Union[str, bytes]:
    encoders = [
        # we don't need to check for dependency as many of those have already been required by NNI
        json_tricks.pathlib_encode,         # pathlib is a required dependency for NNI
        json_tricks.pandas_encode,          # pandas is a required dependency
        json_tricks.numpy_encode,           # required
        json_tricks.encoders.enum_instance_encode,
        json_tricks.json_date_time_encode,  # same as json_tricks
        json_tricks.json_complex_encode,
        json_tricks.json_set_encode,
        json_tricks.numeric_types_encode,
        functools.partial(_json_tricks_serializable_object_encode, use_trace=use_trace),
        _json_tricks_customize_encode,      # After serializable object
        functools.partial(_json_tricks_func_or_cls_encode, pickle_size_limit=pickle_size_limit),
        functools.partial(_json_tricks_any_object_encode, pickle_size_limit=pickle_size_limit),
    ]

    json_tricks_kwargs['allow_nan'] = allow_nan

    if fp is not None:
        return json_tricks.dump(obj, fp, obj_encoders=encoders, **json_tricks_kwargs)
    else:
        return json_tricks.dumps(obj, obj_encoders=encoders, **json_tricks_kwargs)


def load(string: Optional[str] = None, *, fp: Optional[Any] = None,
         preserve_order: bool = False, ignore_comments: bool = True, **json_tricks_kwargs) -> Any:
    """
    Load the string or from file, and convert it to a complex data structure.
    At least one of string or fp has to be not none.

    Parameters
    ----------
    string : str
        JSON string to parse. Can be set to none if fp is used.
    fp : str
        File path to load JSON from. Can be set to none if string is used.
    preserve_order : bool
        `json_tricks parameter <https://json-tricks.readthedocs.io/en/latest/#order>`_
        to use ``OrderedDict`` instead of ``dict``.
        The order is in fact always preserved even when this is False.
    ignore_comments : bool
        Remove comments (starting with ``#`` or ``//``). Default is true.

    Returns
    -------
    any
        The loaded object.
    """
    assert string is not None or fp is not None
    # see encoders for explanation
    hooks = [
        json_tricks.pathlib_hook,
        json_tricks.pandas_hook,
        json_tricks.json_numpy_obj_hook,
        json_tricks.decoders.EnumInstanceHook(),
        json_tricks.json_date_time_hook,
        json_tricks.json_complex_hook,
        json_tricks.json_set_hook,
        json_tricks.numeric_types_hook,
        _json_tricks_serializable_object_decode,
        _json_tricks_customize_decode,
        _json_tricks_func_or_cls_decode,
        _json_tricks_any_object_decode
    ]

    # there was an issue that the user code does not accept ordered dict, and 3.7+ dict has guaranteed order
    json_tricks_kwargs['preserve_order'] = preserve_order
    # to bypass a deprecation warning in json-tricks
    json_tricks_kwargs['ignore_comments'] = ignore_comments

    if string is not None:
        if isinstance(string, IOBase):
            raise TypeError(f'Expect a string, found a {string}. If you intend to use a file, use `nni.load(fp=file)`')
        return json_tricks.loads(string, obj_pairs_hooks=hooks, **json_tricks_kwargs)
    else:
        return json_tricks.load(fp, obj_pairs_hooks=hooks, **json_tricks_kwargs)


class PayloadTooLarge(Exception):
    pass