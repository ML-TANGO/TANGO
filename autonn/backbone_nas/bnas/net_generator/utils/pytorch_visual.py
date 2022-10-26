# https://github.com/zheng-ningxin/Pytorch-Visualization
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
import os


class Pytorch_Visual:
    
    def __init__(self, model, data):
        self.model = model
        self.named_layers = dict(model.named_modules())
        # Add a layer name attribute into each submodule of the model
        for name, layer in self.named_layers.items():
            layer.module_name = name
        # data to used to build the topology of the network
        # For example, data, label = next(iter(dataloader))
        self.data = data
        self.hooks = []
        # Following variables are related with the graph
        # Save the tensor/variable nodes
        self.tensors = set() # save the unique id
        self.layers= set()
        self.id2obj = {} # save the mapping relation ship 
        self.forward_edge = {} # save the depedency relation grapgh
        # self.backward_edge = {}
        
        # The functions need to be hooked to build up the graph
        # Variable use the same add operation of Tensor (torch.Tensor.__add__)
        self.func_need_hook = []
        torch_keys = ['flatten', 'squeeze', 'unsqueeze', 'cat']
        for attr in torch_keys:
            self.func_need_hook.append((torch, attr))
        tensor_keys = [ 'view', '__add__', '__iadd__', 'flatten', 'mean', 'reshape',
                        'squeeze', 'unsqueeze', 'permute', 'contiguous']
        #tensor_keys = ['view', '__add__']
        for attr in tensor_keys:
            self.func_need_hook.append((torch.Tensor, attr))
        # elu is for the activations such as, relu, celu
        # pool is for the pooling layers, such as avg_pool2d 
        functional_keys = [ 'pool', 'batch_norm', 'dropout', 'elu', 
                            'softmax', 'tanh' 'sigmoid', 'interpolate']
        for attr, func in torch.nn.functional.__dict__.items():
            for name_key in functional_keys:
                # filter out the target functional module that need to be hooked
                if name_key in attr:
                    self.func_need_hook.append((torch.nn.functional, attr))
                    break
        self.ori_func = []
        self.visted = set()
        # Init the hook functions
        self.deploy_hooks()
        self.out = self.model(data)
        # Clear the hook functions
        self.remove_hooks()

    @property
    def hooks_length(self):
        return len(self.hooks)
    
    def op_decorator(self, func, name):
        def new_func(*args, **kwargs):

            inputs = []
            if len(args) > 0:
                inputs.extend(self.get_tensors_from(args))
            if len(kwargs) > 0:
                inputs.extend(self.get_tensors_from(kwargs))
            
            out = func(*args, **kwargs)
            # build the graph
            iids = [id(input) for input in inputs]
            oid = id(out)
            self.tensors.update(iids)
            self.tensors.add(oid)
            if oid not in self.id2obj:
                self.id2obj[oid] = out

            for i in range(len(iids)):
                if iids[i] not in self.id2obj:
                    self.id2obj[iids[i]] = inputs[i]
                if iids[i] not in self.forward_edge:
                    self.forward_edge[iids[i]] = [oid]
                elif oid not in self.forward_edge[iids[i]]:
                    self.forward_edge[iids[i]].append(oid)  
            # save the graph related info 
            if hasattr(out, 'graph_info'):
                if 'from' in out.graph_info:
                    # __iadd__ may access a tensor that alreasy exists
                    out.graph_info['from'].append(name)
                else:
                    out.graph_info['from'] = [name]
            else:
                out.graph_info = {'from' : [name]}

            return out
        return new_func


    def get_tensors_from(self, args):
        """
        Some layers/modules may return servaral tensors as output
        (IntermediateLayerGetter) or take multiple tensors as 
        input(torch.cat). Therefore, We need find the real output/input 
        tensor out and build the network architecture based on them.
        Note: 
            I find that the input format of torch.cat is the tuple of tuple
            which looks like ((t1, t2)), so, we use recursive func to
            find all tensors
        """
        tensors = []
        if isinstance(args, dict):
            # some layers may return their output as a dict 
            # ex. the IntermediateLayerGetter in the face detection jobs.
            for key, val in args.items():
                if isinstance(val, torch.Tensor):
                    tensors.append(val)
                else:
                    tensors.extend(self.get_tensors_from(val))
        elif isinstance(args, list) or isinstance(args, tuple):
            # list or tuple
            for item in args:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
                else:
                    tensors.extend(self.get_tensors_from(item))
        elif isinstance(args, torch.Tensor) or isinstance(args, torch.autograd.Variable):
            # if the output is a already a tensor/variable, then return itself
            tensors.append(args)
        return tensors

    
    def get_forward_hook(self):
        def forward_hook(module, inputs, output):

            checker = module.pruneratio_checker
            linputs = list(inputs)
            # Filter the Tensor or Variable inputs out
            linputs = list(filter(lambda x: isinstance(x, Tensor) or isinstance(x, Variable), linputs))
            iids = [id(input) for input in linputs]
            t_outputs = self.get_tensors_from(output)
            oids = [id(t) for t in t_outputs]
            mid = id(module)

            # For the modules that have multiple submodules, for example(Sequential)
            # They will return at here, because the output tensor is already added 
            # by their submodules. Therefore, we only record the lowest level connection.
            flag = False
            for oid in oids:
                if oid not in checker.tensors:
                    flag = True
                    break 
            if not flag:
                # if all output tensors are already created, in this case, the module
                # is a father-module, and we only draw the lowest-level network architecture
                return
            # involve the output tensors into the graph
            self.layers.add(mid)
            self.id2obj[mid] = module
            for tid, _out in enumerate(t_outputs):
                if oids[tid] not in checker.tensors:
                    checker.tensors.add(oids[tid])
                    checker.id2obj[oids[tid]] = _out
                    _out.graph_info = {'from' : [module.module_name]}
            self.forward_edge[mid] = oids

            for i in range(len(iids)):
                if iids[i] not in checker.tensors:
                    checker.tensors.add(iids[i])
                    checker.id2obj[iids[i]] = linputs[i]
                if iids[i] not in checker.forward_edge:
                    checker.forward_edge[iids[i]] = [mid]
                elif mid not in checker.forward_edge[iids[i]]:
                    checker.forward_edge[iids[i]].append(mid)
            # We need to track the input and output tensors from the model perspective
            module.input_tensors = linputs
            module.output_tensor = t_outputs

        return forward_hook


    def deploy_hooks(self):
        # Put the checker's reference into the modules
        # make the graph building easier
        for submodel in self.model.modules():
            submodel.pruneratio_checker = self
        forward_hook = self.get_forward_hook()
        # deploy the hooks
        for submodel in self.model.modules():
            hook_handle = submodel.register_forward_hook(forward_hook)
            self.hooks.append(hook_handle)
        # Hook the tensor/variable operators
        for mod, attr in self.func_need_hook:
            ori_func = getattr(mod, attr)
            self.ori_func.append(ori_func)
            new_func = self.op_decorator(ori_func, attr)
            setattr(mod, attr, new_func)
            
        
    def remove_hooks(self):
        for submodel in self.model.modules():
            if hasattr(submodel, 'pruneratio_checker'):
                delattr(submodel, 'pruneratio_checker')
        for hook in self.hooks:
            hook.remove()
        # reset to the original function
        for i, (mod, attr) in enumerate(self.func_need_hook):
            setattr(mod, attr, self.ori_func[i])
        

    def traverse(self, curid, channel):
        """
            Traverse the tensors and check if the prune ratio
            is legal for the network architecture.
        """

        if curid in self.visted:
            # Only the tensors can be visited twice, the conv layers
            # won't be access twice in the DFS progress
            # check if the dimmension is ok
            if self.id2obj[curid].prune['channel'] != channel:
                return False
            return True
        self.visted.add(curid)
        outchannel = channel
        if isinstance(self.id2obj[curid], torch.Tensor):
            self.id2obj[curid].prune = {'channel' : channel}
        elif isinstance(self.id2obj[curid], torch.nn.Conv2d):
            conv = self.id2obj[curid]
            if hasattr(conv, 'prune'):
                outchannel = int(conv.out_channels * conv.prune['ratio'])
            else:
                outchannel = conv.out_channels
        OK = True

        if curid in self.forward_edge:
            for next_ in self.forward_edge[curid]:
                ## After change the next to next_ , 
                re = self.traverse(next_, outchannel)
                OK = OK and re
        return OK

    def check(self, ratios):
        """
        input:
            ratios: the prune ratios for the layers
            ratios is the dict, in which the keys are 
            the names of the target layer and the values
            are the prune ratio for the corresponding layers
            For example:
            ratios = {'body.conv1': 0.5, 'body.conv2':0.5}
            Note: the name of the layers should looks like 
            the names that model.named_modules() functions 
            returns.
        """
        for name, ratio in ratios.items():
            layer = self.named_layers[name]
            if isinstance(layer, nn.Conv2d):
                layer.prune = {'ratio' : ratio}
            elif isinstance(layer, nn.Linear):
                # Linear usually prune the input direction
                # because the output are often equals to the 
                # the number of the classes in the classification 
                # scenario
                layer.prune = { 'ratio' : ratio }
        self.visted.clear()
        # N * C * H * W
        is_legal = self.traverse(id(self.data), self.data.size(1))
        # remove the ratio tag of the tensor
        for name, ratio in ratios.items():
            layer = self.named_layers[name]
            if hasattr(layer, 'prune'):
                delattr(layer, 'prune')
            if hasattr(layer, 'prune'):
                delattr(layer, 'prune')
        for tid in self.tensors:
            if hasattr(self.id2obj[tid], 'prune'):
                delattr(self.id2obj[tid], 'prune')
        return is_legal
    

    def visual_traverse(self, curid, graph, last_visit):
        """"
        Input:
            graph: the handle of the Dgraph
        """

        if curid in self.visted:
            # Already visited, only connect to the last visited node
            if last_visit is not None:
                graph.edge(str(last_visit), str(curid))
            return
        self.visted.add(curid)
        curobj = self.id2obj[curid]
        if isinstance(curobj, torch.Tensor) or isinstance(curobj, torch.autograd.Variable):
            t_name = 'Tensor:{}'.format(curobj.size())
            if hasattr(curobj, 'graph_info'):
                t_name += '\n Created by %s' % curobj.graph_info['from'] 
            graph.node(str(curid), t_name, shape='box', color='lightblue')
        else:
            graph.node(str(curid), curobj.module_name, shape='ellipse', color='orange')
        # Connect to the last visited node
        if last_visit is not None:
            graph.edge(str(last_visit), str(curid))
        if curid in self.forward_edge:
            for next_ in self.forward_edge[curid]:
                self.visual_traverse(next_, graph, curid)
        

    # def visualization(self, filename='network', format='jpg', debug=False):
    #     """
    #     visualize the network architecture automaticlly.
    #     Input:
    #         filename: the filename of the saved image file
    #         format: the output format
    #         debug: if enable the debug mode
    #     """
    #     import graphviz

    #     graph = graphviz.Digraph(format=format)
    #     self.visted.clear()
    #     graph_start = id(self.data)
    #     self.visual_traverse(graph_start, graph, None)
    #     if debug:
    #         # If enable debug, draw all available tensors in the same
    #         # graph. It the network's architecture are seperated into
    #         # two parts, we can easily and quickly find where the graph
    #         # is broken, and find the missing hook point. 
    #         for tid in self.tensors:
    #             if tid not in self.visted:
    #                 self.visual_traverse(tid, graph, None)
    #     graph.render(filename)
    #     os.remove(filename)
            
if __name__ == '__main__':
    import torchvision.models as models
    model = models.mobilenet_v2()
    data=torch.rand(1,3,224,224)
    pc = Pytorch_Visual(model, data)
    pc.visualization('mobilev2',format='pdf')