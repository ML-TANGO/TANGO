"""
high level support for doing this and that.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class CNode:
    """A dummy docstring."""
    def __init__(self,  # pylint: disable-msg=too-many-arguments, line-too-long
                 id_, type_,
                 params=None, learned_params=None,
                 status=True, group=False):
        self.id_ = id_
        self.type_ = type_
        self.status = status
        self.setparams(params)
        self.setlearnedparams(learned_params)
        self.group = group

    def setparams(self, params):
        """A dummy docstring."""
        assert isinstance(params)
        self.params = self.typecast(params)

    def setlearnedparams(self, learned_params):
        """A dummy docstring."""
        assert isinstance(learned_params)
        self.learned_params = learned_params

    def typecast(self, params):
        """A dummy docstring."""
        datatype = {'in_channels': int, 'out_channels': int,
                    'kernel_size': int, 'stride': int,
                    'padding': int, 'bias': bool,
                    'num_features': int, 'in_features': int,
                    'out_features': int, 'p': float,
                    'dilation': int, 'groups': int,
                    'padding_mode': str, 'eps': float,
                    'momentum': float, 'affine': bool,
                    'track_running_stats': bool, 'return_indices': bool,
                    'ceil_mode': bool, 'count_include_pad': bool,
                    'inplace': bool, 'dim': int, 'output_size': int,
                    'value': float, 'negative_slope': float,
                    'device': type(None), 'dtype': type(None),
                    'weight': type(None), 'size_average': bool,
                    'reduce': bool, 'reduction': str,
                    'ignore_index': type(None), 'label_smoothing': float,
                    'start_dim': int, 'end_dim': int, 'size': type(None),
                    'scale_factor': type(None), 'mode': str,
                    'align_corners': type(None),
                    'recompute_scale_factor': type(None)}

        for key, value in params.items():
            cast = datatype.get(key)
#             print(key, "-->", cast)
            if isinstance(value) is None:
                # use the None as is or use default
                params[key] = value
            elif isinstance(value) == tuple:
                params[key] = tuple(map(cast, value))
            elif isinstance(value) == list:
                params[key] = tuple(map(cast, value))
            elif isinstance(value) == dict:
                if key == "subgraph":
                    for nodeid, param in value.items():
                        param = self.typecast(param)
                        value.update({nodeid: param})
#                         print(nodeId, param)
            else:
                params[key] = cast(value)

        return params

    def getdetails(self):
        """A dummy docstring."""
        print("--------------")
        print(self.type_)
        print(self.learned_params)


class CEdge:  # pylint: disable=too-few-public-methods
    """A dummy docstring."""
    def __init__(self, source, sink, status=True):
        self.source = source
        self.sink = sink
        self.status = status
