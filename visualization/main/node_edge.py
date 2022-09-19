from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import numpy as np  # type: ignore

class c_Node(object):
    def __init__(self, id, type, params={}, learned_params={}, status=True, group=False, **kwargs):
        self.id = id
        self.type = type
        self.status = status
        self.setParams(params)
        self.setLearnedParams(learned_params)
        self.group = group

    def setParams(self, params):
        assert type(params) == type(dict())
        self.params = self.typeCast(params)

    def setLearnedParams(self, learned_params):
        assert type(learned_params) == type(dict())
        self.learned_params = learned_params


    def typeCast(self, params):
        dataType = {'in_channels': int, 'out_channels': int, 'kernel_size': int,
                    'stride': int, 'padding': int, 'bias': bool,
                    'num_features': int, 'in_features': int, 'out_features': int,
                    'p': float, 'dilation': int, 'groups': int,
                    'padding_mode': str, 'eps': float, 'momentum': float,
                    'affine': bool, 'track_running_stats':bool, 'return_indices':bool,
                    'ceil_mode':bool, 'count_include_pad': bool, 'inplace': bool,
                    'dim': int, 'output_size': int, 'value': float, 'negative_slope': float,
                    'device': type(None), 'dtype': type(None), 'weight': type(None),
                    'size_average': bool, 'reduce': bool, 'reduction': str,
                    'ignore_index': type(None), 'label_smoothing': float,
                    'start_dim': int, 'end_dim': int, 'size': type(None), 'scale_factor': type(None),
                    'mode': str, 'align_corners': type(None), 'recompute_scale_factor': type(None)}

        for key, value in params.items():
            cast = dataType.get(key)
#             print(key, "-->", cast)
            if type(value) == type(None):
                # use the None as is or use default
                params[key] = value
            elif type(value) == tuple:
                params[key] = tuple(map(cast, value))
            elif type(value) == list:
                params[key] = tuple(map(cast, value))
            elif type(value) == dict:
                if key == "subgraph":
                    for nodeId, param in value.items():
                        param = self.typeCast(param)
                        value.update({nodeId : param})
#                         print(nodeId, param)
            else:
                params[key] = cast(value)

        return params

    def getDetails(self):
        print("--------------")
        print(self.type)
        print(self.learned_params)

class c_Edge(object):
    def __init__(self, source, sink, status=True):
        self.source = source
        self.sink = sink
        self.status = status

