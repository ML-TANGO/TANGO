'''
search config
'''

SEARCH_SPACE = {
    'kernel_size': [3, 5, 7],
    'expand_ratio': [3, 4, 6],
    # 'stride' : [1, 2],
    'depths': [2, 3, 4],
    'bottleneck_num': 14,
    'population_num': 20,
    'max_generations': 2,
    'tournament_portion': 0.5
}

SEARCH_SPACE['states'] = \
    [len(SEARCH_SPACE['kernel_size'])] * SEARCH_SPACE['bottleneck_num']
