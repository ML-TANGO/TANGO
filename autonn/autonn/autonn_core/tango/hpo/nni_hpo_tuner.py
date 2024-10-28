from nni.experiment import Experiment
import nni

import logging
logger = logging.getLogger(__name__)

def get_tuner_config(args):
    """
    args['tuner_name']에 따라 적절한 튜너 설정을 반환합니다.
    기본값으로 'BOHB' 튜너를 사용합니다.
    """
    tuner_name = args.get('tuner_name', 'BOHB')
    if tuner_name == 'TPE':
        return {
            'name': 'TPE',
            'class_args': {
                'optimize_mode': 'maximize',
                'seed': 12345,
                'tpe_args': {
                    'constant_liar_type': 'mean',
                    'n_startup_jobs': 10,
                    'n_ei_candidates': 20,
                    'linear_forgetting': 100,
                    'prior_weight': 0,
                    'gamma': 0.5
                }
            }
        }
    elif tuner_name == 'BOHB':
        return {
            'name': 'BOHB',
            'class_args': {
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
        }
    elif tuner_name == 'Evolution':
        return {
            'name': 'Evolution',
            'class_args': {
                'optimize_mode': 'maximize',
                'population_size': 100
            }
        }
    elif tuner_name == 'GP':
        return {
            'name': 'GP',
            'class_args': {
                'optimize_mode': 'maximize',
                'utility': 'ei',
                'kappa': 5.0,
                'xi': 0.0,
                'nu': 2.5,
                'alpha': 1e-6,
                'cold_start_num': 10,
                'selection_num_warm_up': 100000,
                'selection_num_starting_points': 250
            }
        }
    elif tuner_name == 'Random':
        return {
            'name': 'Random',
            'class_args': {'optimize_mode': 'maximize'}
        }
    elif tuner_name == 'SMAC':
        return {
            'name': 'SMAC',
            'class_args': {'optimize_mode': 'maximize'}
        }
    elif tuner_name == 'Hyperband':
        return {
            'name': 'Hyperband',
            'class_args': {
                'optimize_mode': 'maximize',
                'R': 60,
                'eta': 3
            }
        }
    elif tuner_name == 'Metis':
        return {
            'name': 'Metis',
            'class_args': {'optimize_mode': 'maximize'}
        }
    else:
        # raise ValueError(f"알 수 없는 튜너 이름입니다: {tuner_name}")
        logger.warning(f"Unknown hpo tuner: {tuner_name}")

def run_experiment(hparams, search_space, args, port=8102):
    experiment = Experiment('local')
    experiment.config.trial_command = 'search.py'
    experiment.config.trial_code_directory = '.'
    experiment.config.search_space = search_space

    # 튜너 설정을 args['tuner_name']에 따라 가져옵니다
    tuner_config = get_tuner_config(args)
    experiment.config.tuner.name = tuner_config['name']
    experiment.config.tuner.class_args = tuner_config['class_args']
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 40

    logger.info(f"run the tuner: {tuner_config['name']}")

    experiment.run(port)
    experiment.stop()

    return hparams