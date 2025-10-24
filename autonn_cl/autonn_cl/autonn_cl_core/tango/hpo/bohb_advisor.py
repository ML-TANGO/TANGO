import os
import sys
import math
from pathlib import Path
from threading import Event, Thread
from queue import Queue, Empty
from typing import Any, Callable, ClassVar
from collections import defaultdict
from enum import Enum

from schema import Schema, Optional, And
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.read_and_write import pcs_new

from tango.hpo.config_generator import CG_BOHB
from tango.utils.hpo_utils import (
    OptimizeMode, 
    Recoverable, 
    Parameters, 
    dump,
    load,
    PayloadTooLarge,
)

import logging
logger = logging.getLogger(__name__)

_next_parameter_id = 0
_KEY = 'TRIAL_BUDGET'
_epsilon = 1e-6

def create_parameter_id():
    """Create an id

    Returns
    -------
    int
        parameter id
    """
    global _next_parameter_id
    _next_parameter_id += 1
    return _next_parameter_id - 1

def create_bracket_parameter_id(brackets_id, brackets_curr_decay, increased_id=-1):
    """Create a full id for a specific bracket's hyperparameter configuration

    Parameters
    ----------
    brackets_id: int
        brackets id
    brackets_curr_decay: int
        brackets curr decay
    increased_id: int
        increased id
    Returns
    -------
    int
        params id
    """
    if increased_id == -1:
        increased_id = str(create_parameter_id())
    params_id = '_'.join([str(brackets_id),
                          str(brackets_curr_decay),
                          increased_id])
    return params_id

class Bracket:
    """
    A bracket in BOHB, all the information of a bracket is managed by
    an instance of this class.

    Parameters
    ----------
    s: int
        The current Successive Halving iteration index.
    s_max: int
        total number of Successive Halving iterations
    eta: float
        In each iteration, a complete run of sequential halving is executed. In it,
		after evaluating each configuration on the same subset size, only a fraction of
		1/eta of them 'advances' to the next round.
	max_budget : float
		The largest budget to consider. Needs to be larger than min_budget!
		The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    """
    def __init__(self, s, s_max, eta, max_budget, optimize_mode):
        self.s = s
        self.s_max = s_max
        self.eta = eta
        self.max_budget = max_budget
        self.optimize_mode = OptimizeMode(optimize_mode)

        self.n = math.ceil((s_max + 1) * eta**s / (s + 1) - _epsilon)
        self.r = max_budget / eta**s
        self.i = 0
        self.hyper_configs = []         # [ {id: params}, {}, ... ]
        self.configs_perf = []          # [ {id: [seq, acc]}, {}, ... ]
        self.num_configs_to_run = []    # [ n, n, n, ... ]
        self.num_finished_configs = []  # [ n, n, n, ... ]
        self.no_more_trial = False

    def is_completed(self):
        """check whether this bracket has sent out all the hyperparameter configurations"""
        return self.no_more_trial

    def get_n_r(self):
        """return the values of n and r for the next round"""
        return math.floor(self.n / self.eta**self.i + _epsilon), math.floor(self.r * self.eta**self.i +_epsilon)

    def increase_i(self):
        """i means the ith round. Increase i by 1"""
        self.i += 1

    def set_config_perf(self, i, parameter_id, seq, value):
        """update trial's latest result with its sequence number, e.g., epoch number or batch number

        Parameters
        ----------
        i: int
            the ith round
        parameter_id: int
            the id of the trial/parameter
        seq: int
            sequence number, e.g., epoch number or batch number
        value: int
            latest result with sequence number seq

        Returns
        -------
        None
        """
        if parameter_id in self.configs_perf[i]:
            if self.configs_perf[i][parameter_id][0] < seq:
                self.configs_perf[i][parameter_id] = [seq, value]
        else:
            self.configs_perf[i][parameter_id] = [seq, value]

    def inform_trial_end(self, i):
        """If the trial is finished and the corresponding round (i.e., i) has all its trials finished,
        it will choose the top k trials for the next round (i.e., i+1)

        Parameters
        ----------
        i: int
            the ith round

        Returns
        -------
        new trial or None:
            If we have generated new trials after this trial end, we will return a new trial parameters.
            Otherwise, we will return None.
        """
        global _KEY
        self.num_finished_configs[i] += 1
        logger.debug('bracket id: %d, round: %d %d, finished: %d, all: %d',
                     self.s, self.i, i, self.num_finished_configs[i], self.num_configs_to_run[i])
        if self.num_finished_configs[i] >= self.num_configs_to_run[i] and self.no_more_trial is False:
            # choose candidate configs from finished configs to run in the next round
            assert self.i == i + 1
            # finish this bracket
            if self.i > self.s:
                self.no_more_trial = True
                return None
            this_round_perf = self.configs_perf[i]
            if self.optimize_mode is OptimizeMode.Maximize:
                sorted_perf = sorted(this_round_perf.items(
                ), key=lambda kv: kv[1][1], reverse=True)  # reverse
            else:
                sorted_perf = sorted(
                    this_round_perf.items(), key=lambda kv: kv[1][1])
            logger.debug(
                'bracket %s next round %s, sorted hyper configs: %s', self.s, self.i, sorted_perf)
            next_n, next_r = self.get_n_r()
            logger.debug('bracket %s next round %s, next_n=%d, next_r=%d',
                         self.s, self.i, next_n, next_r)
            hyper_configs = dict()
            for k in range(next_n):
                params_id = sorted_perf[k][0]
                params = self.hyper_configs[i][params_id]
                params[_KEY] = next_r  # modify r
                # generate new id
                increased_id = params_id.split('_')[-1]
                new_id = create_bracket_parameter_id(
                    self.s, self.i, increased_id)
                hyper_configs[new_id] = params
            self._record_hyper_configs(hyper_configs)
            return [[key, value] for key, value in hyper_configs.items()]
        return None

    def get_hyperparameter_configurations(self, num, r, config_generator):
        """generate num hyperparameter configurations from search space using Bayesian optimization

        Parameters
        ----------
        num: int
            the number of hyperparameter configurations

        Returns
        -------
        list
            a list of hyperparameter configurations. Format: [[key1, value1], [key2, value2], ...]
        """
        global _KEY
        assert self.i == 0
        hyperparameter_configs = dict()
        for _ in range(num):
            params_id = create_bracket_parameter_id(self.s, self.i)
            params = config_generator.get_config(r)
            params[_KEY] = r
            hyperparameter_configs[params_id] = params
        self._record_hyper_configs(hyperparameter_configs)
        return [[key, value] for key, value in hyperparameter_configs.items()]

    def _record_hyper_configs(self, hyper_configs):
        """after generating one round of hyperconfigs, this function records the generated hyperconfigs,
        creates a dict to record the performance when those hyperconifgs are running, set the number of finished configs
        in this round to be 0, and increase the round number.

        Parameters
        ----------
        hyper_configs: list
            the generated hyperconfigs
        """
        self.hyper_configs.append(hyper_configs)
        self.configs_perf.append(dict())
        self.num_finished_configs.append(0)
        self.num_configs_to_run.append(len(hyper_configs))
        self.increase_i()

class ClassArgsValidator(object):
    """
    NNI tuners/assessors/adivisors accept a `classArgs` parameter in experiment configuration file.
    This ClassArgsValidator interface is used to validate the classArgs section in exeperiment
    configuration file.
    """
    def validate_class_args(self, **kwargs):
        """
        Validate the classArgs configuration in experiment configuration file.

        Parameters
        ----------
        kwargs: dict
            kwargs passed to tuner/assessor/advisor constructor

        Raises:
            Raise an execption if the kwargs is invalid.
        """
        pass

    def choices(self, key, *args):
        """
        Utility method to create a scheme to check whether the `key` is one of the `args`.

        Parameters:
        ----------
        key: str
            key name of the data to be validated
        args: list of str
            list of the choices

        Returns: Schema
        --------
            A scheme to check whether the `key` is one of the `args`.
        """
        return And(lambda n: n in args, error='%s should be in [%s]!' % (key, str(args)))

    def range(self, key, keyType, start, end):
        """
        Utility method to create a schema to check whether the `key` is in the range of [start, end].

        Parameters:
        ----------
        key: str
            key name of the data to be validated
        keyType: type
            python data type, such as int, float
        start: type is specified by keyType
            start of the range
        end: type is specified by keyType
            end of the range

        Returns: Schema
        --------
            A scheme to check whether the `key` is in the range of [start, end].
        """
        return And(
            And(keyType, error='%s should be %s type!' % (key, keyType.__name__)),
            And(lambda n: start <= n <= end, error='%s should be in range of (%s, %s)!' % (key, start, end))
        )

    def path(self, key):
        return And(
            And(str, error='%s should be a string!' % key),
            And(lambda p: Path(p).exists(), error='%s path does not exist!' % (key))
        )

class BOHBClassArgsValidator(ClassArgsValidator):
    def validate_class_args(self, **kwargs):
        Schema({
            'optimize_mode': self.choices('optimize_mode', 'maximize', 'minimize'),
            Optional('min_budget'): self.range('min_budget', int, 0, 9999),
            Optional('max_budget'): self.range('max_budget', int, 0, 9999),
            Optional('eta'): self.range('eta', int, 0, 9999),
            Optional('min_points_in_model'): self.range('min_points_in_model', int, 0, 9999),
            Optional('top_n_percent'): self.range('top_n_percent', int, 1, 99),
            Optional('num_samples'): self.range('num_samples', int, 1, 9999),
            Optional('random_fraction'): self.range('random_fraction', float, 0, 9999),
            Optional('bandwidth_factor'): self.range('bandwidth_factor', float, 0, 9999),
            Optional('min_bandwidth'): self.range('min_bandwidth', float, 0, 9999),
            Optional('config_space'): self.path('config_space')
        }).validate(kwargs)

from collections import namedtuple

_trial_env_var_names = [
    'NNI_PLATFORM',
    'NNI_EXP_ID',
    'NNI_TRIAL_JOB_ID',
    'NNI_SYS_DIR',
    'NNI_OUTPUT_DIR',
    'NNI_TRIAL_COMMAND_CHANNEL',
    'NNI_TRIAL_SEQ_ID',
    'MULTI_PHASE',
    'REUSE_MODE',
]

_dispatcher_env_var_names = [
    'SDK_PROCESS',
    'NNI_MODE',
    'NNI_CHECKPOINT_DIRECTORY',
    'NNI_LOG_DIRECTORY',
    'NNI_LOG_LEVEL',
    'NNI_INCLUDE_INTERMEDIATE_RESULTS',
    'NNI_TUNER_COMMAND_CHANNEL',
]

def _load_env_vars(env_var_names):
    env_var_dict = {k: os.environ.get(k) for k in env_var_names}
    return namedtuple('EnvVars', env_var_names)(**env_var_dict)  # pylint: disable=unused-variable

trial_env_vars = _load_env_vars(_trial_env_var_names)

dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)

class CommandType(Enum):
    # in
    Initialize = 'IN'
    RequestTrialJobs = 'GE'
    ReportMetricData = 'ME'
    UpdateSearchSpace = 'SS'
    ImportData = 'FD'
    AddCustomizedTrialJob = 'AD'
    TrialEnd = 'EN'
    Terminate = 'TE'
    Ping = 'PI'

    # out
    Initialized = 'ID'
    NewTrialJob = 'TR'
    SendTrialJobParameter = 'SP'
    NoMoreTrialJobs = 'NO'
    KillTrialJob = 'KI'
    Error = 'ER'

class TunerIncomingCommand:
    # For type checking.
    command_type: ClassVar[CommandType]

class TunerCommandChannel:
    """
    A channel to communicate with NNI manager.

    Each NNI experiment has a channel URL for tuner/assessor/strategy algorithm.
    The channel can only be connected once, so for each Python side :class:`~nni.experiment.Experiment` object,
    there should be exactly one corresponding ``TunerCommandChannel`` instance.

    :meth:`connect` must be invoked before sending or receiving data.

    The constructor does not have side effect so ``TunerCommandChannel`` can be created anywhere.
    But :meth:`connect` requires an initialized NNI manager, or otherwise the behavior is unpredictable.

    :meth:`_send` and :meth:`_receive` are underscore-prefixed because their signatures are scheduled to change by v3.0.

    Parameters
    ----------
    url
        The command channel URL.
        For now it must be like ``"ws://localhost:8080/tuner"`` or ``"ws://localhost:8080/url-prefix/tuner"``.
    """

    def __init__(self, url: str):
        self._channel = WsChannelClient(url)
        self._callbacks: dict[CommandType, list[Callable[..., None]]] = defaultdict(list)

    def connect(self) -> None:
        self._channel.connect()

    def disconnect(self) -> None:
        self._channel.disconnect()

    def listen(self, stop_event: Event) -> None:
        """Listen for incoming commands.

        Call :meth:`receive` in a loop and call ``callback`` for each command,
        until ``stop_event`` is set, or a Terminate command is received.
        All commands will go into callback, including Terminate command.

        It usually runs in a separate thread.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`TunerIncomingCommand` as argument.
            It's not expected to return anything.
        stop_event
            A threading event that can be used to stop the loop.
        """
        while not stop_event.is_set():
            received = self.receive()
            for callback in self._callbacks[received.command_type]:
                callback(received)

            # Two ways to stop the loop:
            # 1. The received command is a Terminate command, which is triggered by a NNI manager stop.
            # 2. The stop_event is set from another thread (possibly main thread), which could be an engine shutdown.
            if received.command_type == CommandType.Terminate:
                logger.debug('Received command type is terminate. Stop listening.')
                stop_event.set()

    # NOTE: The semantic commands are only partial for the convenience of NAS implementation.
    # Send commands are broken into different functions and signatures.
    # Ideally it should be similar for receive commands, but we can't predict which command will appear in receive.

    def send_initialized(self) -> None:
        """Send an initialized command to NNI manager."""
        self._send(CommandType.Initialized, '')

    def send_trial(
        self,
        parameter_id: int,
        parameters: Parameters,
        parameter_source: str = 'algorithm',
        parameter_index: int = 0,
        placement_constraint: dict[str, Any] | None = None,  # TODO: Define PlacementConstraint class.
    ):
        """
        Send a new trial job to NNI manager.

        Without multi-phase in mind, one parameter = one trial.

        Parameters
        ----------
        parameter_id
            The ID of the current parameter.
            It's used by whoever calls the :meth:`send_trial` function to identify the parameters.
            In most cases, they are non-negative integers starting from 0.
        parameters
            The parameters.
        parameter_source
            The source of the parameters. ``algorithm`` means the parameters are generated by the algorithm.
            It should be left as default in most cases.
        parameter_index
            The index of the parameters. This is previously used in multi-phase, but now it's only kept for compatibility reasons.
        placement_constraint
            The placement constraint of the created trial job.
        """
        # Local import to reduce import delay.
        from nni.common.version import version_dump

        trial_dict = {
            'parameter_id': parameter_id,
            'parameters': parameters,
            'parameter_source': parameter_source,
            'parameter_index': parameter_index,
            'version_info': version_dump()
        }
        if placement_constraint is not None:
            _validate_placement_constraint(placement_constraint)
            trial_dict['placement_constraint'] = placement_constraint

        try:
            send_payload = dump(trial_dict, pickle_size_limit=int(os.getenv('PICKLE_SIZE_LIMIT', 64 * 1024)))
        except PayloadTooLarge:
            raise ValueError(
                'Serialization failed when trying to dump the model because payload too large (larger than 64 KB). '
                'This is usually caused by pickling large objects (like datasets) by mistake. '
                'See the full error traceback for details and https://nni.readthedocs.io/en/stable/NAS/Serialization.html '
                'for how to resolve such issue. '
            )

        self._send(CommandType.NewTrialJob, send_payload)

    def send_no_more_trial_jobs(self) -> None:
        """Tell NNI manager that there are no more trial jobs to send for now."""
        self._send(CommandType.NoMoreTrialJobs, '')

    def receive(self) -> TunerIncomingCommand:
        """Receives a command from NNI manager."""
        command_type, data = self._receive()
        if data:
            data = load(data)

        # NOTE: Only handles the commands that are used by NAS.
        # It uses somewhat hacky way to convert the data received from NNI manager
        # to a semantic command.
        if command_type is None:
            # This shouldn't happen. Only for robustness.
            _logger.warning('Received command is empty. Terminating...')
            return Terminate()
        elif command_type == CommandType.Terminate:
            return Terminate()
        elif command_type == CommandType.Initialize:
            if not isinstance(data, dict):
                raise TypeError(f'Initialize command data must be a dict, but got {type(data)}')
            return Initialize(data)
        elif command_type == CommandType.RequestTrialJobs:
            if not isinstance(data, int):
                raise TypeError(f'RequestTrialJobs command data must be an integer, but got {type(data)}')
            return RequestTrialJobs(data)
        elif command_type == CommandType.UpdateSearchSpace:
            if not isinstance(data, dict):
                raise TypeError(f'UpdateSearchSpace command data must be a dict, but got {type(data)}')
            return UpdateSearchSpace(data)
        elif command_type == CommandType.ReportMetricData:
            if not isinstance(data, dict):
                raise TypeError(f'ReportMetricData command data must be a dict, but got {type(data)}')
            if 'value' in data:
                data['value'] = load(data['value'])
            return ReportMetricData(**data)
        elif command_type == CommandType.TrialEnd:
            if not isinstance(data, dict):
                raise TypeError(f'TrialEnd command data must be a dict, but got {type(data)}')
            # For some reason, only one parameter (I guess the first one) shows up in the data.
            # But a trial technically is associated with multiple parameters.
            parameter_id = load(data['hyper_params'])['parameter_id']
            return TrialEnd(
                trial_job_id=data['trial_job_id'],
                parameter_ids=[parameter_id],
                event=data['event']
            )
        else:
            raise ValueError(f'Unknown command type: {command_type}')

    def on_terminate(self, callback: Callable[[Terminate], None]) -> None:
        """Register a callback for Terminate command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`Terminate` as argument.
        """
        self._callbacks[Terminate.command_type].append(callback)

    def on_initialize(self, callback: Callable[[Initialize], None]) -> None:
        """Register a callback for Initialize command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`Initialize` as argument.
        """
        self._callbacks[Initialize.command_type].append(callback)

    def on_request_trial_jobs(self, callback: Callable[[RequestTrialJobs], None]) -> None:
        """Register a callback for RequestTrialJobs command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`RequestTrialJobs` as argument.
        """
        self._callbacks[RequestTrialJobs.command_type].append(callback)

    def on_update_search_space(self, callback: Callable[[UpdateSearchSpace], None]) -> None:
        """Register a callback for UpdateSearchSpace command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`UpdateSearchSpace` as argument.
        """
        self._callbacks[UpdateSearchSpace.command_type].append(callback)

    def on_report_metric_data(self, callback: Callable[[ReportMetricData], None]) -> None:
        """Register a callback for ReportMetricData command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`ReportMetricData` as argument.
        """
        self._callbacks[ReportMetricData.command_type].append(callback)

    def on_trial_end(self, callback: Callable[[TrialEnd], None]) -> None:
        """Register a callback for TrialEnd command.

        Parameters
        ----------
        callback
            A callback function that takes a :class:`TrialEnd` as argument.
        """
        self._callbacks[TrialEnd.command_type].append(callback)

    def _send(self, command_type: CommandType, data: str) -> None:
        self._channel.send({'type': command_type.value, 'content': data})

    def _receive(self) -> tuple[CommandType, str] | tuple[None, None]:
        command = self._channel.receive()
        if command is None:
            return None, None
        else:
            return CommandType(command['type']), command.get('content', '')

def _validate_placement_constraint(placement_constraint):
    # Currently only for CGO.
    if placement_constraint is None:
        raise ValueError('placement_constraint is None')
    if not 'type' in placement_constraint:
        raise ValueError('placement_constraint must have `type`')
    if not 'gpus' in placement_constraint:
        raise ValueError('placement_constraint must have `gpus`')
    if placement_constraint['type'] not in ['None', 'GPUNumber', 'Device']:
        raise ValueError('placement_constraint.type must be either `None`,. `GPUNumber` or `Device`')
    if placement_constraint['type'] == 'None' and len(placement_constraint['gpus']) > 0:
        raise ValueError('placement_constraint.gpus must be an empty list when type == None')
    if placement_constraint['type'] == 'GPUNumber':
        if len(placement_constraint['gpus']) != 1:
            raise ValueError('placement_constraint.gpus currently only support one host when type == GPUNumber')
        for e in placement_constraint['gpus']:
            if not isinstance(e, int):
                raise ValueError('placement_constraint.gpus must be a list of number when type == GPUNumber')
    if placement_constraint['type'] == 'Device':
        for e in placement_constraint['gpus']:
            if not isinstance(e, tuple):
                raise ValueError('placement_constraint.gpus must be a list of tuple when type == Device')
            if not (len(e) == 2 and isinstance(e[0], str) and isinstance(e[1], int)):
                raise ValueError('placement_constraint.gpus`s tuple must be (str, int)')

class MsgDispatcherBase(Recoverable):
    """
    This is where tuners and assessors are not defined yet.
    Inherits this class to make your own advisor.

    .. note::

        The class inheriting MsgDispatcherBase should be instantiated
        after nnimanager (rest server) is started, so that the object
        is ready to use right after its instantiation.
    """

    def __init__(self, command_channel_url=None):
        super().__init__()
        self.stopping = False
        if command_channel_url is None:
            command_channel_url = dispatcher_env_vars.NNI_TUNER_COMMAND_CHANNEL
        self._channel = TunerCommandChannel(command_channel_url)
        # NOTE: `connect()` should be put in __init__. First, this `connect()` affects nnimanager's
        # starting process, without `connect()` nnimanager is blocked in `dispatcher.init()`.
        # Second, nas experiment uses a thread to execute `run()` of this class, thus, there is
        # no way to know when the websocket between nnimanager and dispatcher is built. The following
        # logic may crash is websocket is not built. One example is updating search space. If updating
        # search space too soon, as the websocket has not been built, the rest api of updating search
        # space will timeout.
        # FIXME: this is making unittest happy
        if not command_channel_url.startswith('ws://_unittest_'):
            self._channel.connect()
        self.default_command_queue = Queue()
        self.assessor_command_queue = Queue()
        # here daemon should be True, because their parent thread is configured as daemon to enable smooth exit of NAS experiment.
        # if daemon is not set, these threads will block the daemon effect of their parent thread.
        self.default_worker = Thread(target=self.command_queue_worker, args=(self.default_command_queue,), daemon=True)
        self.assessor_worker = Thread(target=self.command_queue_worker, args=(self.assessor_command_queue,), daemon=True)
        self.worker_exceptions = []

    def run(self):
        """Run the tuner.
        This function will never return unless raise.
        """
        logger.info('Dispatcher started')

        self.default_worker.start()
        self.assessor_worker.start()

        if dispatcher_env_vars.NNI_MODE == 'resume':
            self.load_checkpoint()

        while not self.stopping:
            command, data = self._channel._receive()
            if data:
                data = load(data)

            if command is None or command is CommandType.Terminate:
                break
            self.enqueue_command(command, data)
            if self.worker_exceptions:
                break

        _logger.info('Dispatcher exiting...')
        self.stopping = True
        self.default_worker.join()
        self.assessor_worker.join()
        self._channel.disconnect()

        _logger.info('Dispatcher terminiated')

    def report_error(self, error: str) -> None:
        '''
        Report dispatcher error to NNI manager.
        '''
        _logger.info(f'Report error to NNI manager: {error}')
        try:
            self.send(CommandType.Error, error)
        except Exception:
            _logger.error('Connection to NNI manager is broken. Failed to report error.')

    def send(self, command, data):
        self._channel._send(command, data)

    def command_queue_worker(self, command_queue):
        """Process commands in command queues.
        """
        while True:
            try:
                # set timeout to ensure self.stopping is checked periodically
                command, data = command_queue.get(timeout=3)
                try:
                    self.process_command(command, data)
                except Exception as e:
                    _logger.exception(e)
                    self.worker_exceptions.append(e)
                    break
            except Empty:
                pass
            if self.stopping and (_worker_fast_exit_on_terminate or command_queue.empty()):
                break

    def enqueue_command(self, command, data):
        """Enqueue command into command queues
        """
        if command == CommandType.TrialEnd or (
                command == CommandType.ReportMetricData and data['type'] == 'PERIODICAL'):
            self.assessor_command_queue.put((command, data))
        else:
            self.default_command_queue.put((command, data))

        qsize = self.default_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('default queue length: %d', qsize)

        qsize = self.assessor_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            _logger.warning('assessor queue length: %d', qsize)

    def process_command(self, command, data):
        _logger.debug('process_command: command: [%s], data: [%s]', command, data)

        command_handlers = {
            # Tuner commands:
            CommandType.Initialize: self.handle_initialize,
            CommandType.RequestTrialJobs: self.handle_request_trial_jobs,
            CommandType.UpdateSearchSpace: self.handle_update_search_space,
            CommandType.ImportData: self.handle_import_data,
            CommandType.AddCustomizedTrialJob: self.handle_add_customized_trial,

            # Tuner/Assessor commands:
            CommandType.ReportMetricData: self.handle_report_metric_data,

            CommandType.TrialEnd: self.handle_trial_end,
            CommandType.Ping: self.handle_ping,
        }
        if command not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))
        command_handlers[command](data)

    def handle_ping(self, data):
        pass

    def handle_initialize(self, data):
        """Initialize search space and tuner, if any
        This method is meant to be called only once for each experiment, after calling this method,
        dispatcher should `send(CommandType.Initialized, '')`, to set the status of the experiment to be "INITIALIZED".
        Parameters
        ----------
        data: dict
            search space
        """
        raise NotImplementedError('handle_initialize not implemented')

    def handle_request_trial_jobs(self, data):
        """The message dispatcher is demanded to generate ``data`` trial jobs.
        These trial jobs should be sent via ``send(CommandType.NewTrialJob, nni.dump(parameter))``,
        where ``parameter`` will be received by NNI Manager and eventually accessible to trial jobs as "next parameter".
        Semantically, message dispatcher should do this ``send`` exactly ``data`` times.

        The JSON sent by this method should follow the format of

        ::

            {
                "parameter_id": 42
                "parameters": {
                    // this will be received by trial
                },
                "parameter_source": "algorithm" // optional
            }

        Parameters
        ----------
        data: int
            number of trial jobs
        """
        raise NotImplementedError('handle_request_trial_jobs not implemented')

    def handle_update_search_space(self, data):
        """This method will be called when search space is updated.
        It's recommended to call this method in `handle_initialize` to initialize search space.
        *No need to* notify NNI Manager when this update is done.
        Parameters
        ----------
        data: dict
            search space
        """
        raise NotImplementedError('handle_update_search_space not implemented')

    def handle_import_data(self, data):
        """Import previous data when experiment is resumed.
        Parameters
        ----------
        data: list
            a list of dictionaries, each of which has at least two keys, 'parameter' and 'value'
        """
        raise NotImplementedError('handle_import_data not implemented')

    def handle_add_customized_trial(self, data):
        """Experimental API. Not recommended for usage.
        """
        raise NotImplementedError('handle_add_customized_trial not implemented')

    def handle_report_metric_data(self, data):
        """Called when metric data is reported or new parameters are requested (for multiphase).
        When new parameters are requested, this method should send a new parameter.

        Parameters
        ----------
        data: dict
            a dict which contains 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.
            type: can be `MetricType.REQUEST_PARAMETER`, `MetricType.FINAL` or `MetricType.PERIODICAL`.
            `REQUEST_PARAMETER` is used to request new parameters for multiphase trial job. In this case,
            the dict will contain additional keys: `trial_job_id`, `parameter_index`. Refer to `msg_dispatcher.py`
            as an example.

        Raises
        ------
        ValueError
            Data type is not supported
        """
        raise NotImplementedError('handle_report_metric_data not implemented')

    def handle_trial_end(self, data):
        """Called when the state of one of the trials is changed

        Parameters
        ----------
        data: dict
            a dict with keys: trial_job_id, event, hyper_params.
            trial_job_id: the id generated by training service.
            event: the job’s state.
            hyper_params: the string that is sent by message dispatcher during the creation of trials.

        """
        raise NotImplementedError('handle_trial_end not implemented')

class BOHB(MsgDispatcherBase):
    """
    `BOHB <https://arxiv.org/abs/1807.01774>`__ is a robust and efficient hyperparameter tuning algorithm at scale.
    BO is an abbreviation for "Bayesian Optimization" and HB is an abbreviation for "Hyperband".

    BOHB relies on HB (Hyperband) to determine how many configurations to evaluate with which budget,
    but it replaces the random selection of configurations at the beginning of each HB iteration
    by a model-based search (Bayesian Optimization).
    Once the desired number of configurations for the iteration is reached,
    the standard successive halving procedure is carried out using these configurations.
    It keeps track of the performance of all function evaluations g(x, b) of configurations x
    on all budgets b to use as a basis for our models in later iterations.
    Please refer to the paper :footcite:t:`falkner2018bohb` for detailed algorithm.

    Note that BOHB needs additional installation using the following command:

    .. code-block:: bash

        pip install nni[BOHB]

    Examples
    --------

    .. code-block::

        config.tuner.name = 'BOHB'
        config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'min_budget': 1,
            'max_budget': 27,
            'eta': 3,
            'min_points_in_model': 7,
            'top_n_percent': 15,
            'num_samples': 64,
            'random_fraction': 0.33,
            'bandwidth_factor': 3.0,
            'min_bandwidth': 0.001
        }

    Parameters
    ----------
    optimize_mode: str
        Optimize mode, 'maximize' or 'minimize'.
    min_budget: float
        The smallest budget to assign to a trial job, (budget can be the number of mini-batches or epochs).
        Needs to be positive.
    max_budget: float
        The largest budget to assign to a trial job. Needs to be larger than min_budget.
        The budgets will be geometrically distributed
        :math:`a^2 + b^2 = c^2 \\sim \\eta^k` for :math:`k\\in [0, 1, ... , num\\_subsets - 1]`.
    eta: int
        In each iteration, a complete run of sequential halving is executed. In it,
        after evaluating each configuration on the same subset size, only a fraction of
        1/eta of them 'advances' to the next round.
        Must be greater or equal to 2.
    min_points_in_model: int
        Number of observations to start building a KDE. Default 'None' means dim+1;
        when the number of completed trials in this budget is equal to or larger than ``max{dim+1, min_points_in_model}``,
        BOHB will start to build a KDE model of this budget then use said KDE model to guide configuration selection.
        Needs to be positive. (dim means the number of hyperparameters in search space)
    top_n_percent: int
        Percentage (between 1 and 99, default 15) of the observations which are considered good.
        Good points and bad points are used for building KDE models.
        For example, if you have 100 observed trials and top_n_percent is 15,
        then the top 15% of points will be used for building the good points models "l(x)".
        The remaining 85% of points will be used for building the bad point models "g(x)".
    num_samples: int
        Number of samples to optimize EI (default 64).
        In this case, it will sample "num_samples" points and compare the result of l(x)/g(x).
        Then it will return the one with the maximum l(x)/g(x) value as the next configuration
        if the optimize_mode is ``maximize``. Otherwise, it returns the smallest one.
    random_fraction: float
        Fraction of purely random configurations that are sampled from the prior without the model.
    bandwidth_factor: float
        To encourage diversity, the points proposed to optimize EI are sampled
        from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3).
        It is suggested to use the default value if you are not familiar with KDE.
    min_bandwidth: float
        To keep diversity, even when all (good) samples have the same value for one of the parameters,
        a minimum bandwidth (default: 1e-3) is used instead of zero.
        It is suggested to use the default value if you are not familiar with KDE.
    config_space: str
        Directly use a .pcs file serialized by `ConfigSpace <https://automl.github.io/ConfigSpace/>` in "pcs new" format.
        In this case, search space file (if provided in config) will be ignored.
        Note that this path needs to be an absolute path. Relative path is currently not supported.

    Notes
    -----

    Below is the introduction of the BOHB process separated in two parts:

    **The first part HB (Hyperband).**
    BOHB follows Hyperband’s way of choosing the budgets and continue to use SuccessiveHalving.
    For more details, you can refer to the :class:`nni.algorithms.hpo.hyperband_advisor.Hyperband`
    and the `reference paper for Hyperband <https://arxiv.org/abs/1603.06560>`__.
    This procedure is summarized by the pseudocode below.

    .. image:: ../../img/bohb_1.png
        :scale: 80 %
        :align: center

    **The second part BO (Bayesian Optimization)**
    The BO part of BOHB closely resembles TPE with one major difference:
    It opted for a single multidimensional KDE compared to the hierarchy of one-dimensional KDEs used in TPE
    in order to better handle interaction effects in the input space.
    Tree Parzen Estimator(TPE): uses a KDE (kernel density estimator) to model the densities.

    .. image:: ../../img/bohb_2.png
        :scale: 80 %
        :align: center

    To fit useful KDEs, we require a minimum number of data points Nmin;
    this is set to d + 1 for our experiments, where d is the number of hyperparameters.
    To build a model as early as possible, we do not wait until Nb = \|Db\|,
    where the number of observations for budget b is large enough to satisfy q · Nb ≥ Nmin.
    Instead, after initializing with Nmin + 2 random configurations, we choose the
    best and worst configurations, respectively, to model the two densities.
    Note that it also samples a constant fraction named **random fraction** of the configurations uniformly at random.

    .. image:: ../../img/bohb_3.png
        :scale: 80 %
        :align: center


    .. image:: ../../img/bohb_6.jpg
        :scale: 65 %
        :align: center

    **The above image shows the workflow of BOHB.**
    Here set max_budget = 9, min_budget = 1, eta = 3, others as default.
    In this case, s_max = 2, so we will continuously run the {s=2, s=1, s=0, s=2, s=1, s=0, ...} cycle.
    In each stage of SuccessiveHalving (the orange box), it will pick the top 1/eta configurations and run them again with more budget,
    repeating the SuccessiveHalving stage until the end of this iteration.
    At the same time, it collects the configurations, budgets and final metrics of each trial
    and use these to build a multidimensional KDEmodel with the key "budget".
    Multidimensional KDE is used to guide the selection of configurations for the next iteration.
    The sampling procedure (using Multidimensional KDE to guide selection) is summarized by the pseudocode below.

    .. image:: ../../img/bohb_4.png
        :scale: 80 %
        :align: center

    **Here is a simple experiment which tunes MNIST with BOHB.**
    Code implementation: :githublink:`examples/trials/mnist-advisor <examples/trials/mnist-advisor>`
    The following is the experimental final results:

    .. image:: ../../img/bohb_5.png
        :scale: 80 %
        :align: center

    More experimental results can be found in the `reference paper <https://arxiv.org/abs/1807.01774>`__.
    It shows that BOHB makes good use of previous results and has a balanced trade-off in exploration and exploitation.
    """

    def __init__(self,
                 optimize_mode='maximize',
                 min_budget=1,
                 max_budget=3,
                 eta=3,
                 min_points_in_model=None,
                 top_n_percent=15,
                 num_samples=64,
                 random_fraction=1/3,
                 bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 config_space=None):
        super(BOHB, self).__init__()
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.min_points_in_model = min_points_in_model
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction
        self.bandwidth_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.config_space = config_space

        # all the configs waiting for run
        self.generated_hyper_configs = []
        # all the completed configs
        self.completed_hyper_configs = []

        self.s_max = math.floor(
            math.log(self.max_budget / self.min_budget, self.eta) + _epsilon)
        # current bracket(s) number
        self.curr_s = self.s_max
        # In this case, tuner increases self.credit to issue a trial config sometime later.
        self.credit = 0
        self.brackets = dict()
        self.search_space = None
        # [key, value] = [parameter_id, parameter]
        self.parameters = dict()

        # config generator
        self.cg = None

        # record the latest parameter_id of the trial job trial_job_id.
        # if there is no running parameter_id, self.job_id_para_id_map[trial_job_id] == None
        # new trial job is added to this dict and finished trial job is removed from it.
        self.job_id_para_id_map = dict()
        # record the unsatisfied parameter request from trial jobs
        self.unsatisfied_jobs = []

    def handle_initialize(self, data):
        """Initialize Tuner, including creating Bayesian optimization-based parametric models
        and search space formations

        Parameters
        ----------
        data: search space
            search space of this experiment

        Raises
        ------
        ValueError
            Error: Search space is None
        """
        logger.info('start to handle_initialize')
        # convert search space jason to ConfigSpace
        self.handle_update_search_space(data)

        # generate BOHB config_generator using Bayesian optimization
        if self.search_space:
            self.cg = CG_BOHB(configspace=self.search_space,
                              min_points_in_model=self.min_points_in_model,
                              top_n_percent=self.top_n_percent,
                              num_samples=self.num_samples,
                              random_fraction=self.random_fraction,
                              bandwidth_factor=self.bandwidth_factor,
                              min_bandwidth=self.min_bandwidth)
        else:
            raise ValueError('Error: Search space is None')
        # generate first brackets
        self.generate_new_bracket()
        self.send(CommandType.Initialized, '')

    def generate_new_bracket(self):
        """generate a new bracket"""
        logger.debug(
            'start to create a new SuccessiveHalving iteration, self.curr_s=%d', self.curr_s)
        if self.curr_s < 0:
            logger.info("s < 0, Finish this round of Hyperband in BOHB. Generate new round")
            self.curr_s = self.s_max
        self.brackets[self.curr_s] = Bracket(
            s=self.curr_s, s_max=self.s_max, eta=self.eta,
            max_budget=self.max_budget, optimize_mode=self.optimize_mode
        )
        next_n, next_r = self.brackets[self.curr_s].get_n_r()
        logger.debug(
            'new SuccessiveHalving iteration, next_n=%d, next_r=%d', next_n, next_r)
        # rewrite with TPE
        generated_hyper_configs = self.brackets[self.curr_s].get_hyperparameter_configurations(
            next_n, next_r, self.cg)
        self.generated_hyper_configs = generated_hyper_configs.copy()

    def handle_request_trial_jobs(self, data):
        """recerive the number of request and generate trials

        Parameters
        ----------
        data: int
            number of trial jobs that nni manager ask to generate
        """
        # Receive new request
        self.credit += data

        for _ in range(self.credit):
            self._request_one_trial_job()

    def _get_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration.

        If this function is called, Command will be sent by BOHB:
        a. If there is a parameter need to run, will return "NewTrialJob" with a dict:
        {
            'parameter_id': id of new hyperparameter
            'parameter_source': 'algorithm'
            'parameters': value of new hyperparameter
        }
        b. If BOHB don't have parameter waiting, will return "NoMoreTrialJobs" with
        {
            'parameter_id': '-1_0_0',
            'parameter_source': 'algorithm',
            'parameters': ''
        }
        """
        if not self.generated_hyper_configs:
            ret = {
                'parameter_id': '-1_0_0',
                'parameter_source': 'algorithm',
                'parameters': ''
            }
            self.send(CommandType.NoMoreTrialJobs, nni.dump(ret))
            return None
        assert self.generated_hyper_configs
        params = self.generated_hyper_configs.pop(0)
        ret = {
            'parameter_id': params[0],
            'parameter_source': 'algorithm',
            'parameters': params[1]
        }
        self.parameters[params[0]] = params[1]
        return ret

    def _request_one_trial_job(self):
        """get one trial job, i.e., one hyperparameter configuration.

        If this function is called, Command will be sent by BOHB:
        a. If there is a parameter need to run, will return "NewTrialJob" with a dict:
        {
            'parameter_id': id of new hyperparameter
            'parameter_source': 'algorithm'
            'parameters': value of new hyperparameter
        }
        b. If BOHB don't have parameter waiting, will return "NoMoreTrialJobs" with
        {
            'parameter_id': '-1_0_0',
            'parameter_source': 'algorithm',
            'parameters': ''
        }
        """
        ret = self._get_one_trial_job()
        if ret is not None:
            self.send(CommandType.NewTrialJob, nni.dump(ret))
            self.credit -= 1

    def handle_update_search_space(self, data):
        """change json format to ConfigSpace format dict<dict> -> configspace

        Parameters
        ----------
        data: JSON object
            search space of this experiment
        """
        search_space = data
        cs = None
        logger.debug(f'Received data: {data}')
        if self.config_space:
            logger.info(f'Got a ConfigSpace file path, parsing the search space directly from {self.config_space}. '
                        'The NNI search space is ignored.')
            with open(self.config_space, 'r') as fh:
                cs = pcs_new.read(fh)
        else:
            cs = CS.ConfigurationSpace()
            for var in search_space:
                _type = str(search_space[var]["_type"])
                if _type == 'choice':
                    cs.add_hyperparameter(CSH.CategoricalHyperparameter(
                        var, choices=search_space[var]["_value"]))
                elif _type == 'randint':
                    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1] - 1))
                elif _type == 'uniform':
                    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1]))
                elif _type == 'quniform':
                    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1],
                        q=search_space[var]["_value"][2]))
                elif _type == 'loguniform':
                    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1],
                        log=True))
                elif _type == 'qloguniform':
                    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(
                        var, lower=search_space[var]["_value"][0], upper=search_space[var]["_value"][1],
                        q=search_space[var]["_value"][2], log=True))
                elif _type == 'normal':
                    cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                        var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2]))
                elif _type == 'qnormal':
                    cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                        var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2],
                        q=search_space[var]["_value"][3]))
                elif _type == 'lognormal':
                    cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                        var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2],
                        log=True))
                elif _type == 'qlognormal':
                    cs.add_hyperparameter(CSH.NormalFloatHyperparameter(
                        var, mu=search_space[var]["_value"][1], sigma=search_space[var]["_value"][2],
                        q=search_space[var]["_value"][3], log=True))
                else:
                    raise ValueError(
                        'unrecognized type in search_space, type is {}'.format(_type))

        self.search_space = cs

    def handle_trial_end(self, data):
        """receive the information of trial end and generate next configuaration.

        Parameters
        ----------
        data: dict()
            it has three keys: trial_job_id, event, hyper_params
            trial_job_id: the id generated by training service
            event: the job's state
            hyper_params: the hyperparameters (a string) generated and returned by tuner
        """
        hyper_params = nni.load(data['hyper_params'])
        if self.is_created_in_previous_exp(hyper_params['parameter_id']):
            # The end of the recovered trial is ignored
            return
        logger.debug('Tuner handle trial end, result is %s', data)
        self._handle_trial_end(hyper_params['parameter_id'])
        if data['trial_job_id'] in self.job_id_para_id_map:
            del self.job_id_para_id_map[data['trial_job_id']]

    def _send_new_trial(self):
        while self.unsatisfied_jobs:
            ret = self._get_one_trial_job()
            if ret is None:
                break
            one_unsatisfied = self.unsatisfied_jobs.pop(0)
            ret['trial_job_id'] = one_unsatisfied['trial_job_id']
            ret['parameter_index'] = one_unsatisfied['parameter_index']
            # update parameter_id in self.job_id_para_id_map
            self.job_id_para_id_map[ret['trial_job_id']] = ret['parameter_id']
            self.send(CommandType.SendTrialJobParameter, nni.dump(ret))
        for _ in range(self.credit):
            self._request_one_trial_job()

    def _handle_trial_end(self, parameter_id):
        s, i, _ = parameter_id.split('_')
        hyper_configs = self.brackets[int(s)].inform_trial_end(int(i))

        if hyper_configs is not None:
            logger.debug(
                'bracket %s next round %s, hyper_configs: %s', s, i, hyper_configs)
            self.generated_hyper_configs = self.generated_hyper_configs + hyper_configs
        # Finish this bracket and generate a new bracket
        elif self.brackets[int(s)].no_more_trial:
            self.curr_s -= 1
            self.generate_new_bracket()
        self._send_new_trial()

    def handle_report_metric_data(self, data):
        """reveice the metric data and update Bayesian optimization with final result

        Parameters
        ----------
        data:
            it is an object which has keys 'parameter_id', 'value', 'trial_job_id', 'type', 'sequence'.

        Raises
        ------
        ValueError
            Data type not supported
        """
        if self.is_created_in_previous_exp(data['parameter_id']):
            if data['type'] == MetricType.FINAL:
                # only deal with final metric using import data
                param = self.get_previous_param(data['parameter_id'])
                trial_data = [{'parameter': param, 'value': nni.load(data['value'])}]
                self.handle_import_data(trial_data)
            return
        logger.debug('handle report metric data = %s', data)
        if 'value' in data:
            data['value'] = nni.load(data['value'])
        if data['type'] == MetricType.REQUEST_PARAMETER:
            assert multi_phase_enabled()
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            assert data['trial_job_id'] in self.job_id_para_id_map
            self._handle_trial_end(self.job_id_para_id_map[data['trial_job_id']])
            ret = self._get_one_trial_job()
            if ret is None:
                self.unsatisfied_jobs.append({'trial_job_id': data['trial_job_id'], 'parameter_index': data['parameter_index']})
            else:
                ret['trial_job_id'] = data['trial_job_id']
                ret['parameter_index'] = data['parameter_index']
                # update parameter_id in self.job_id_para_id_map
                self.job_id_para_id_map[data['trial_job_id']] = ret['parameter_id']
                self.send(CommandType.SendTrialJobParameter, nni.dump(ret))
        else:
            assert 'value' in data
            value = extract_scalar_reward(data['value'])
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -value
            else:
                reward = value
            assert 'parameter_id' in data
            s, i, _ = data['parameter_id'].split('_')
            logger.debug('bracket id = %s, metrics value = %s, type = %s', s, value, data['type'])
            s = int(s)

            # add <trial_job_id, parameter_id> to self.job_id_para_id_map here,
            # because when the first parameter_id is created, trial_job_id is not known yet.
            if data['trial_job_id'] in self.job_id_para_id_map:
                assert self.job_id_para_id_map[data['trial_job_id']] == data['parameter_id']
            else:
                self.job_id_para_id_map[data['trial_job_id']] = data['parameter_id']

            assert 'type' in data
            if data['type'] == MetricType.FINAL:
                # and PERIODICAL metric are independent, thus, not comparable.
                assert 'sequence' in data
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], sys.maxsize, value)
                self.completed_hyper_configs.append(data)

                _parameters = self.parameters[data['parameter_id']]
                _parameters.pop(_KEY)
                # update BO with loss, max_s budget, hyperparameters
                self.cg.new_result(loss=reward, budget=data['sequence'], parameters=_parameters, update_model=True)
            elif data['type'] == MetricType.PERIODICAL:
                self.brackets[s].set_config_perf(
                    int(i), data['parameter_id'], data['sequence'], value)
            else:
                raise ValueError(
                    'Data type not supported: {}'.format(data['type']))

    def handle_add_customized_trial(self, data):
        global _next_parameter_id
        # data: parameters
        previous_max_param_id = self.recover_parameter_id(data)
        _next_parameter_id = previous_max_param_id + 1

    def handle_import_data(self, data):
        """Import additional data for tuning

        Parameters
        ----------
        data:
            a list of dictionarys, each of which has at least two keys, 'parameter' and 'value'

        Raises
        ------
        AssertionError
            data doesn't have required key 'parameter' and 'value'
        """
        for entry in data:
            entry['value'] = nni.load(entry['value'])
        _completed_num = 0
        for trial_info in data:
            logger.info("Importing data, current processing progress %s / %s", _completed_num, len(data))
            _completed_num += 1
            assert "parameter" in trial_info
            _params = trial_info["parameter"]
            assert "value" in trial_info
            _value = trial_info['value']
            if not _value:
                logger.info("Useless trial data, value is %s, skip this trial data.", _value)
                continue
            _value = extract_scalar_reward(_value)
            budget_exist_flag = False
            barely_params = dict()
            for keys in _params:
                if keys == _KEY:
                    _budget = _params[keys]
                    budget_exist_flag = True
                else:
                    barely_params[keys] = _params[keys]
            if not budget_exist_flag:
                _budget = self.max_budget
                logger.info("Set \"TRIAL_BUDGET\" value to %s (max budget)", self.max_budget)
            if self.optimize_mode is OptimizeMode.Maximize:
                reward = -_value
            else:
                reward = _value
            self.cg.new_result(loss=reward, budget=_budget, parameters=barely_params, update_model=True)
        logger.info("Successfully import tuning data to BOHB advisor.")
