"""
high level support for doing this and that.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from tkinter import messagebox
from copy import deepcopy


# adadjacencylist: 연결되었든, 안되었든 노드들의 ID와 연결정보 담고 있음.
# ex) 1: [3] -> 1번 노드와 3번 노드가 연결되어있음.
#     5: [] -> 5번노드는 아무데도 연결되어있지 않음, 만약 병렬구조면 1:[3,5]
# activeedgelist: adadjacencylist중, 연결된 노드들의 ID와 연결정보만 담고 있음
# nodes: 노드들의 모든 정보를 담고 있는 리스트
# (클래스 형태로 id, type, params, learned_params, status를 담고 있음)
class CGraph:
    """A dummy docstring."""
    def __init__(self, graph=None, adadjacencylist=None,
                 activeedgelist=None, nodes=None):
        if graph:
            self.adadjacencylist = deepcopy(graph.adadjacencylist)
            self.activeedgelist = deepcopy(graph.activeedgelist)
            self.nodes = deepcopy(graph.nodes)
        elif adadjacencylist and activeedgelist and nodes:
            self.adadjacencylist = deepcopy(adadjacencylist)
            self.activeedgelist = deepcopy(activeedgelist)
            self.nodes = deepcopy(nodes)
        else:
            self.adadjacencylist = {}
            self.activeedgelist = {}
            self.nodes = {}

    # def display(self):
    #     print("=============== GRAPH =================")
    #     n = len(self.adadjacencylist)
    #     if n == 0:
    #         print("Empty Graph")
    #     else:
    #         for key, val in self.adadjacencylist.items():
    #             node = self.nodes.get(key) # check if active
    #             if node.status == False:
    #                 continue
    #             print(key, "->", val)
    #     print("=======================================")

    def __str__(self):
        return str(self.getparamlist())

#     def __repr__(self):
#         out = {}
#         for node in self.nodes.values():
#             out.update({node.id : node.params})
#
#         return str(out)

    def getparamlist(self):
        """A dummy docstring."""
        out = []
        for node in self.nodes.values():
            out.append({node: node.params})

        return out

    def addnode(self, node):
        """A dummy docstring."""
        if not self.adadjacencylist.get(node.id_, None):
            self.nodes.update({node.id_: node})
            self.adadjacencylist.update({node.id_: []})
        else:
            print('Node already exists')

    def editnode(self, _id,  # pylint: disable-msg=too-many-arguments
                 _type=None, params=None, status=True):
        """A dummy docstring."""
        node = self.nodes.get(_id)
        if _type:
            node.type = _type
        if params:
            try:
                node.setparams(params)
            except BaseException:  # pylint: disable-msg=broad-except
                # log("Invalid Parameters - params are not dictionary")
                messagebox.showerror('Invalid Parameters',
                                     "Please enter parameters as "
                                     + "a valid dictionary."
                                     + "\ne.g. {'kernel_size': (2, 2), "
                                     + "'stride': (2, 2), 'padding': (0, 0)}")

        if status == 'setactive':
            node.status = True
        elif status == 'setinactive':
            node.status = False

        # if group == True:
        #     node.group = True
        # elif group == False:
        #     node.group = False
        self.nodes.update({_id: node})

    def deletenode(self, _id):
        """A dummy docstring."""
        del self.nodes[_id]
        if _id in self.adadjacencylist:
            del self.adadjacencylist[_id]
        if _id in self.activeedgelist:
            del self.activeedgelist[_id]

    # 선으로 이었을 때 실행됨
    def addedge(self, edge):
        """A dummy docstring."""
        # 1. get existing list for the source node
        adjacencies = self.adadjacencylist.get(edge.source)
        # 2. add new node (sink node) to the list
        adjacencies.append(edge.sink)
        # 3. update the list
        self.adadjacencylist.update({edge.source: adjacencies})
        self.activeedgelist.update({edge.source: adjacencies})

    def editedge(self, edge, activity=True):
        """A dummy docstring."""
        activelist = self.activeedgelist.get(edge.source)
        if activity:
            if edge.sink in activelist:
                activelist.remove(edge.sink)  # make inactive
        else:
            if edge.sink not in activelist:
                activelist.append(edge.sink)  # make active
        self.activeedgelist.update({edge.source: activelist})

    def deleteedge(self, source, sink):
        """A dummy docstring."""
        # 1. get existing list for the source node
        adjacencies = self.adadjacencylist.get(source)
        activelist = self.activeedgelist.get(source)

        if not adjacencies:  # return if called after deletenode
            return
        if sink in adjacencies:
            adjacencies.remove(sink)
        self.adadjacencylist.update({source: adjacencies})

        if not activelist:
            return
        if sink in activelist:
            activelist.remove(sink)
        self.activeedgelist.update({source: activelist})

    def topological_sort_util(self, _nodeid, visited, stack):
        """A dummy docstring."""
        # Mark the current node as visited.
        visited.update({_nodeid: True})
        # Recur for all the vertices adjacent to this vertex
        for node_id in self.adadjacencylist.get(_nodeid):
            if visited.get(node_id) is False:
                self.topological_sort_util(node_id, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, _nodeid)

    # The function to do Topological Sort. It uses recursive
    # topologicalsortutil()
    # sort는 display를 위한 것. 우린 안써도됨.
    def topological_sort(self):
        """A dummy docstring."""
        stack = []
        e_stack = {}
        visited = {}
        for key, val in self.adadjacencylist.items():
            node = self.nodes.get(key)  # check if active
            if node.status is False:
                continue
            visited.update({key: False})

        for key, val in self.adadjacencylist.items():
            if visited.get(key) is False:
                self.topological_sort_util(key, visited, stack)

        for value in stack:
            e_stack[value] = {'prior': [], 'next': []}
            e_stack[value]['next'] = self.adadjacencylist[value]
            for key, val in self.adadjacencylist.items():
                if value in val:
                    e_stack[value]['prior'].append(key)

        return stack, e_stack

    # Toolbolx.py에서 컴파일할 때 사용함
    def normalize(self):
        # pylint: disable=too-many-locals, too-many-branches
        """A dummy docstring."""
        id_count = 0
        deleted_nodes = []
        convert_dict = {}
        topo, expanded = self.topological_sort()
        # pylint: disable=too-many-nested-blocks, missing-class-docstring
        for _id in list(topo):
            if self.nodes.get(_id).type == 'PASS':
                self.deletenode(_id)
                deleted_nodes.append(_id)
                for node in self.adadjacencylist:
                    if _id in self.adadjacencylist[node]:
                        self.adadjacencylist[node].remove(_id)
                for node in self.activeedgelist:
                    if _id in self.activeedgelist[node]:
                        self.activeedgelist[node].remove(_id)
                expanded_prior = expanded[_id]['prior']
                expanded_next = expanded[_id]['next']
                for p_id in expanded_prior:
                    if p_id not in deleted_nodes:
                        for e_id in expanded_next:
                            if e_id not in deleted_nodes:
                                if p_id in self.adadjacencylist and \
                                        e_id not in self.adadjacencylist[p_id]:
                                    self.adadjacencylist[p_id].append(e_id)
                                if p_id in self.activeedgelist and \
                                        e_id not in self.activeedgelist[p_id]:
                                    self.activeedgelist[p_id].append(e_id)
                topo.remove(_id)
                continue
            id_count += 1
            convert_dict[_id] = id_count
        old_nodes = dict(self.nodes)
        old_adj = dict(self.adadjacencylist)
        old_active = dict(self.activeedgelist)
        self.nodes = {}
        self.adadjacencylist = {}
        self.activeedgelist = {}
        for _id in old_nodes:
            self.nodes[convert_dict[_id]] = old_nodes[_id]
        for _id in old_adj:
            new_adj_list = []
            old_adj_list = old_adj[_id]
            for item in old_adj_list:
                new_adj_list.append(convert_dict[item])
            self.adadjacencylist[convert_dict[_id]] = list(new_adj_list)
        for _id in old_active:
            new_active_list = []
            old_active_list = old_active[_id]
            for item in old_active_list:
                new_active_list.append(convert_dict[item])
            self.activeedgelist[convert_dict[_id]] = list(new_active_list)
        return self


class CNode:
    """A dummy docstring."""
    # status : 클릭했는가? (setactive/setinactive)
    def __init__(self, id_,  # pylint: disable-msg=too-many-arguments
                 type_, params=None, learned_params=None, status=True):
        self.id_ = id_
        self.type_ = type_
        self.status = status
        self.setparams(params)
        self.setlearnedparams(learned_params)

    def setparams(self, params):
        '''setparams'''
        # assert type(params) == type(dict())
        assert isinstance(params, type({}))
        self.params = self.typecast(params)

    def setlearnedparams(self, learned_params):
        '''setlearnedparams'''
        assert isinstance(learned_params, type({}))
        self.learned_params = learned_params

    def typecast(self, params):
        """A dummy docstring."""
        datatype = {'in_channels': int,
                    'out_channels': int,
                    'kernel_size': int,
                    'stride': int,
                    'padding': int,
                    'bias': bool,
                    'num_features': int,
                    'in_features': int,
                    'out_features': int,
                    'p': float,
                    'dilation': int,
                    'groups': int,
                    'padding_mode': str,
                    'eps': float,
                    'momentum': float,
                    'affine': bool,
                    'track_running_stats': bool,
                    'return_indices': bool,
                    'ceil_mode': bool,
                    'count_include_pad': bool,
                    'inplace': bool,
                    'dim': int,
                    'output_size': int,
                    'value': float,
                    'negative_slope': float,
                    'device': type(None),
                    'dtype': type(None),
                    'weight': type(None),
                    'size_average': bool,
                    'reduce': bool,
                    'reduction': str,
                    'ignore_index': type(None),
                    'label_smoothing': float,
                    'start_dim': int,
                    'end_dim': int,
                    'size': type(None),
                    'scale_factor': type(None),
                    'mode': str,
                    'align_corners': type(None),
                    'recompute_scale_factor': type(None)}

        for key, value in params.items():
            cast = datatype.get(key)
            if isinstance(value, type(None)):
                # use the None as is or use default
                params[key] = value
            elif isinstance(value, type(tuple)):
                params[key] = tuple(map(cast, value))
            elif isinstance(value, type(list)):
                params[key] = tuple(map(cast, value))
            elif isinstance(value, type(dict)):
                if key == "subgraph":
                    for _nodeid, param in value.items():
                        param = self.typecast(param)
                        value.update({_nodeid: param})
            else:
                params[key] = cast(value)

        return params

    def getdetails(self):
        """A dummy docstring."""
        print("--------------")
        print(self.type_)
        print(self.learned_params)


class CEdge:  # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    def __init__(self, source, sink, status=True):
        self.source = source
        self.sink = sink
        self.status = status


class CTest():  # pylint: disable-msg=too-few-public-methods
    """A dummy docstring."""
    def test_branches(self):
        """A dummy docstring."""
        graph = CGraph()
        graph.addnode(CNode("conv1", type_="Conv2D"))
        graph.addnode(CNode("maxpool1", type_="MaxPool2D"))
        graph.addnode(CNode("bn1", type_="BatchNorm2D"))
        graph.addnode(CNode("conv2a", type_="Conv2D"))
        graph.addnode(CNode("conv2b", type_="Conv2D"))
        graph.addnode(CNode("relu", type_="ReLU"))
        graph.addnode(CNode("maxpool2", type_="MaxPool2D"))

        graph.addedge(CEdge("conv1", "maxpool1"))
        graph.addedge(CEdge("maxpool1", "bn1"))
        graph.addedge(CEdge("bn1", "conv2a"))
        graph.addedge(CEdge("bn1", "conv2b"))
        graph.addedge(CEdge("conv2a", "relu"))
        graph.addedge(CEdge("conv2b", "relu"))
        graph.addedge(CEdge("relu", "maxpool2"))
        graph_obj = CGraph(graph).normalize()
        order = graph_obj.topological_sort()

        print(order)
        return order


class CShow2():
    """A dummy docstring."""
    def __str__(self):
        self.test_branches()
        print("order")
        return str(0)

    def test_branches(self):
        """A dummy docstring."""
        graph = CGraph()
        graph.addnode(CNode("conv1", type_="Conv2D",
                            params={'nIn': 1, 'nOut': 6, 'kW': 3, 'kH': 3}))
        graph.addnode(CNode("maxpool1", type_="MaxPool2D", status=False))
        graph.addnode(CNode("bn1", type_="BatchNorm2D"))
        graph.addnode(CNode("conv2a", type_="Conv2D",
                            params={'nIn': 6, 'nOut': 6, 'kW': 3, 'kH': 3}))
        graph.addnode(CNode("conv2b", type_="Conv2D",
                            params={'nIn': 6, 'nOut': 6, 'kW': 1, 'kH': 1}))
        graph.addnode(CNode("relu", type_="ReLU"))
        graph.addnode(CNode("maxpool2", type_="MaxPool2D"))
        graph.addedge(CEdge("conv1", "maxpool1"))
        graph.addedge(CEdge("maxpool1", "bn1"))
        graph.addedge(CEdge("bn1", "conv2a"))
        graph.addedge(CEdge("bn1", "conv2b"))
        graph.addedge(CEdge("conv2a", "relu"))
        graph.addedge(CEdge("conv2b", "relu"))
        graph.addedge(CEdge("relu", "maxpool2"))
        order = graph.topological_sort()
        print(order)
        return order
