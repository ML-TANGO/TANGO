from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numbers
from six import text_type, integer_types, binary_type
import numpy as np  # type: ignore

#from Log import *
# from Binder import *
from tkinter import messagebox
from copy import copy, deepcopy

#from node_edge import Edge
#from node_edge import Node


class c_Graph(object):
    # adjacencyList : 연결되었든, 안되었든 노드들의 ID와 연결정보 담고 있음. ex) 1 : [3] -> 1번 노드와 3번 노드가 연결되어있음. 5 : [] -> 5번노드는 아무데도 연결되어있지 않음, 만약 병렬구조면 1:[3,5]
    # activeEdgeList : adjacencyList중, 연결된 노드들의 ID와 연결정보만 담고 있음
    # nodes : 노드들의 모든 정보를 담고 있는 리스트 (클래스 형태로 id, type, params, learned_params, status를 담고 있음)
    def __init__(self, graph=None, adjacencyList=None, activeEdgeList=None, nodes=None):
        if graph:
            self.adjacencyList = deepcopy(graph.adjacencyList)
            self.activeEdgeList = deepcopy(graph.activeEdgeList)
            self.nodes = deepcopy(graph.nodes)
        elif adjacencyList and activeEdgeList and nodes:
            self.adjacencyList = deepcopy(adjacencyList)
            self.activeEdgeList = deepcopy(activeEdgeList)
            self.nodes = deepcopy(nodes)
        else:
            self.adjacencyList = {}
            self.activeEdgeList = {}
            self.nodes = {}

    # def display(self):
    #     print("=============== GRAPH =================")
    #     n = len(self.adjacencyList)
    #     if n == 0:
    #         print("Empty Graph")
    #     else:
    #         for key, val in self.adjacencyList.items():
    #             node = self.nodes.get(key) # check if active
    #             if node.status == False:
    #                 continue
    #             print(key, "->", val)
    #     print("=======================================")

    def __str__(self):
        return self.getParamList()

#     def __repr__(self):
#         out = {}
#         for node in self.nodes.values():
#             out.update({node.id : node.params})
#
#         return str(out)

    def getParamList(self):
        out = []
        for node in self.nodes.values():
            out.append({node : node.params})

        return out

    def addNode(self, node):
        if not self.adjacencyList.get(node.id, None):
            self.nodes.update({node.id: node})
            self.adjacencyList.update({node.id : []})
        else:
            print('Node already exists')

    def editNode(self, id, type=None, params=None, status=True, group=None):
        node = self.nodes.get(id)
        if type:
            node.type = type
        if params:
            try:
                node.setParams(params)
            except:
                log("Invalid Parameters - params are not dictionary")
                messagebox.showerror('Invalid Parameters', "Please enter parameters as a valid dictionary. \ne.g. {'kernel_size': (2, 2), 'stride': (2, 2), 'padding': (0, 0)}")

        if status == 'setactive':
            node.status = True
        elif status == 'setinactive':
            node.status = False

        # if group == True:
        #     node.group = True
        # elif group == False:
        #     node.group = False
        self.nodes.update({id: node})

    def deleteNode(self, id):
        del self.nodes[id]
        if id in self.adjacencyList:
            del self.adjacencyList[id]
        if id in self.activeEdgeList:
            del self.activeEdgeList[id]

    # 선으로 이었을 때 실행됨
    def addEdge(self, edge):
        #1. get existing list for the source node
        adjacencies = self.adjacencyList.get(edge.source)
        #2. add new node (sink node) to the list
        adjacencies.append(edge.sink)
        #3. update the list
        self.adjacencyList.update({edge.source : adjacencies})
        self.activeEdgeList.update({edge.source : adjacencies})

    def editEdge(self, edge, activity=True):
        activeList = self.activeEdgeList.get(edge.source)
        if activity:
            if edge.sink in activeList:
                activeList.remove(edge.sink) # make inactive
        else:
            if edge.sink not in activeList:
                activeList.append(edge.sink) # make active
        self.activeEdgeList.update({edge.source : activeList})

    def deleteEdge(self, source, sink):
        #1. get existing list for the source node
        adjacencies = self.adjacencyList.get(source)
        activeList = self.activeEdgeList.get(source)

        if not adjacencies: # return if called after deleteNode
            return
        if sink in adjacencies:
            adjacencies.remove(sink)
        self.adjacencyList.update({source : adjacencies})

        if not activeList:
            return
        if sink in activeList:
            activeList.remove(sink)
        self.activeEdgeList.update({source : activeList})


        # A recursive function used by topologicalSort
    def topologicalSortUtil(self, nodeId, visited, stack):
        # Mark the current node as visited.
        visited.update({nodeId : True})
        # Recur for all the vertices adjacent to this vertex
        for node_id in self.adjacencyList.get(nodeId):
            if visited.get(node_id) == False:
                self.topologicalSortUtil(node_id, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, nodeId)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    # sort는 display를 위한 것. 우린 안써도됨.
    def topologicalSort(self):
        stack = []
        expanded_stack = {}
        # Mark all the vertices as not visited
        visited = {}
        for key, val in self.adjacencyList.items():
            node = self.nodes.get(key) # check if active
            if node.status == False:
                continue
            visited.update({key : False})

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for key, val in self.adjacencyList.items():
            if visited.get(key) == False:
                self.topologicalSortUtil(key, visited, stack)

        # expand stack to find dependencies: prior to pull from and next to push to
        for value in stack:
            expanded_stack[value] = {'prior':[], 'next':[]}
            expanded_stack[value]['next'] = self.adjacencyList[value]
            for key, val in self.adjacencyList.items():
                if value in val:
                    expanded_stack[value]['prior'].append(key)

        # return contents of stack as topological sorted order
        return stack, expanded_stack

    # Toolbolx.py에서 컴파일할 때 사용함
    def normalize(self):
        id_count = 0
        deleted_nodes = []
        convert_dict = {}
        topo, expanded = self.topologicalSort()
        for id in list(topo):
            if self.nodes.get(id).type == 'PASS':
                self.deleteNode(id)
                deleted_nodes.append(id)
                for node in self.adjacencyList:
                    if id in self.adjacencyList[node]:
                        self.adjacencyList[node].remove(id)
                for node in self.activeEdgeList:
                    if id in self.activeEdgeList[node]:
                        self.activeEdgeList[node].remove(id)
                expanded_prior = expanded[id]['prior']
                expanded_next = expanded[id]['next']
                for p_id in expanded_prior:  # connect prior to next for each empty group node
                    if p_id not in deleted_nodes:
                        for e_id in expanded_next:
                            if e_id not in deleted_nodes:
                                if p_id in self.adjacencyList and e_id not in self.adjacencyList[p_id]:
                                    self.adjacencyList[p_id].append(e_id)
                                if p_id in self.activeEdgeList and e_id not in self.activeEdgeList[p_id]:
                                    self.activeEdgeList[p_id].append(e_id)
                topo.remove(id)
                continue
            id_count += 1
            convert_dict[id] = id_count
        old_nodes = dict(self.nodes)
        old_adj = dict(self.adjacencyList)
        old_active = dict(self.activeEdgeList)
        self.nodes = {}
        self.adjacencyList = {}
        self.activeEdgeList = {}
        for id in old_nodes:
            self.nodes[convert_dict[id]] = old_nodes[id]
        for id in old_adj:
            new_adj_list = []
            old_adj_list = old_adj[id]
            for item in old_adj_list:
                new_adj_list.append(convert_dict[item])
            self.adjacencyList[convert_dict[id]] = list(new_adj_list)
        for id in old_active:
            new_active_list = []
            old_active_list = old_active[id]
            for item in old_active_list:
                new_active_list.append(convert_dict[item])
            self.activeEdgeList[convert_dict[id]] = list(new_active_list)
        return self


class c_Node(object):
    #status : 클릭했는가? (setactive/setinactive)
    # def __init__(self, id, type, params={}, learned_params={}, status=True, group=False, **kwargs):
    def __init__(self, id, type, params={}, learned_params={}, status=True, **kwargs):
        self.id = id
        self.type = type
        self.status = status
        self.setParams(params)
        self.setLearnedParams(learned_params)
        #self.group = group

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

class c_test():
    def test_branches():
        graph = c_Graph()
        graph.addNode(c_Node("conv1", type="Conv2D"))
        graph.addNode(c_Node("maxpool1", type="MaxPool2D"))
        graph.addNode(c_Node("bn1", type="BatchNorm2D"))
        graph.addNode(c_Node("conv2a", type="Conv2D"))
        graph.addNode(c_Node("conv2b", type="Conv2D"))
        graph.addNode(c_Node("relu", type="ReLU"))
        graph.addNode(c_Node("maxpool2", type="MaxPool2D"))

        graph.addEdge(c_Edge("conv1", "maxpool1"))
        graph.addEdge(c_Edge("maxpool1", "bn1"))
        graph.addEdge(c_Edge("bn1", "conv2a"))
        graph.addEdge(c_Edge("bn1", "conv2b"))
        graph.addEdge(c_Edge("conv2a", "relu"))
        graph.addEdge(c_Edge("conv2b", "relu"))
        graph.addEdge(c_Edge("relu", "maxpool2"))
        graph_obj = c_Graph(graph).normalize()
        order = graph_obj.topologicalSort()

        print(order)
        return order

class c_show2():
    def __str__(self):
        self.test_branches()
        print("order")
        return 0

    def test_branches(self):
        graph = c_Graph()
        graph.addNode(c_Node("conv1", type="Conv2D", params={'nIn':1, 'nOut':6, 'kW':3, 'kH':3}))
        graph.addNode(c_Node("maxpool1", type="MaxPool2D", status = False))
        graph.addNode(c_Node("bn1", type="BatchNorm2D"))
        graph.addNode(c_Node("conv2a", type="Conv2D", params={'nIn':6, 'nOut':6, 'kW':3, 'kH':3}))
        graph.addNode(c_Node("conv2b", type="Conv2D", params={'nIn':6, 'nOut':6, 'kW':1, 'kH':1}))
        graph.addNode(c_Node("relu", type="ReLU"))
        graph.addNode(c_Node("maxpool2", type="MaxPool2D"))

        graph.addEdge(c_Edge("conv1", "maxpool1"))
        graph.addEdge(c_Edge("maxpool1", "bn1"))
        graph.addEdge(c_Edge("bn1", "conv2a"))
        graph.addEdge(c_Edge("bn1", "conv2b"))
        graph.addEdge(c_Edge("conv2a", "relu"))
        graph.addEdge(c_Edge("conv2b", "relu"))
        graph.addEdge(c_Edge("relu", "maxpool2"))
        #graph.display()
        order = graph.topologicalSort()
        print(order)
        #graph.createModel(order)
        return order
