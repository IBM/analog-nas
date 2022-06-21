import argparse
import sys
import re
from typing import Callable, Any, List

import torch
import torchvision.models as models
from torch import Tensor
import torch.nn as nn
import numpy as np 

from utils import * 
from tracer import ModulePathTracer

parser = argparse.ArgumentParser(description='ADS Definition Generation')
parser.add_argument('--model_arch',
                    default='resnet18', type=str,
                    choices=["resnet18", "resnet20", "resnet44", "resnet56", "resnet32", "resnet34", "resnet50"],
                    help='choose architecture among resnet variants')
args = parser.parse_args()

sys.stdout = open('{}.ads'.format(args.model_arch), 'w')

# dictionary to convert layer names from PyTorch to ADS. (contains only resnet operations)
layer_to_ads = {
    "Conv2d" : "CONV", 
    "BatchNorm2d" : "BatchNorm",
    "ReLU" : "RELU", 
    "MaxPool2d": "POOLING",
    "AdaptiveAvgPool2d": "POOLING", 
    "Linear": "IP"
}

"""
Class that holds all the parameters of an ADS layer.
"""
class ADSlayer:
    def __init__(self, name, index, inp, out, type, parameters=None):
        self.name = name
        self.id = index
        self.type = type
        self.inp = inp
        self.out = out 
        self.parameters = parameters

    def set_parameters(self, parameters):
        self.parameters = parameters

    def __repr__(self):
        result = "Lid={},Ltype={}\n".format(self.id, self.type)
        if self.parameters != None:
            for n, p in self.parameters.items():
                if "Dim" in n:
                    #print(p)
                    p = [str(int) for int in p]
                    dim = ",".join(p[1:])
                    result += "\t{}={}:{}\n".format(n, p[0], dim)
                else:
                    result += "\t{}={}\n".format(n, p)
            
        for i in self.inp:
            result += "\tInp={}\n".format(i)
  
        if self.type == "CONV" or self.type == "IP":
            result += "\twPrecision=8\n\tIPrecision=8\n\tOPrecision=8\n\tErrInPrecision=8\n\tErrOutPrecision=8\n"
        else:
            result += "\tIPrecision=8\n\tOPrecision=8\n"
        result += "\tOut={}".format(self.out)
        return result


class HookLayerADSDefinition(nn.Module):
    """
    HookLayerADSDefinition: registers a hook in each layer to extract layer information
    such as kernel size, stride, padding... 
    and writes the corresponding ads definition. 
    """
    def __init__(self, model: nn.Module, inputs_, outputs_):
        super().__init__()
        self.model = model
        #self.inputs = inputs_
        # Converting into list of tuple
        self.inputs  = [(k, v) for k, v in inputs_.items()]
        self.outputs = outputs_
        self.ignore = 0
        self.mvm = 0
        self.modules = {}
        self.inp = ["data"]
        self.not_down = 0
        self.bn2 = 0
        self.id = 1
        self.id_ = 2
        i = 1
        bn2 = 0
        for name, layer in self.model.named_modules():
            name = name.replace(".", "_")
            if list(layer.children()) == []:
                #print(name, type(layer))
                self.modules[name] = layer
                layer.register_forward_hook(self.extract_size_hook(layer, name, self.id))


    def extract_size_hook(self, layer, name, id) -> Callable:
        def fn(layer, input, output):
            #print(name)
            #print(output.shape)
            parameters = {}
            if isinstance(layer, nn.Conv2d):
                weight_dim = [4, output.shape[1], input[0].shape[1], layer.kernel_size[0], layer.kernel_size[1]]
                biasPresent = 0
                if layer.bias != None: 
                    biasPresent = 1
                input_dim = [2, input[0].shape[1], input[0].shape[2]*input[0].shape[3]]
                output_dim = [2, output.shape[1], output.shape[2]*output.shape[3]]
                im2colDim = compute_shape_col(input[0].shape, layer.kernel_size, layer.padding, layer.stride)
                im2col_dim = [2, im2colDim[0], im2colDim[1]]
                stride = layer.stride[0]

                if output.shape[1]> 2000 or im2colDim[0] > 2000 :
                    self.ignore = 1
                self.mvm +=  im2colDim[1]
                parameters = {
                    "weightDim": weight_dim, 
                    "biasPresent": biasPresent, 
                    "inputDim": input_dim, 
                    "outputDim": output_dim, 
                    "im2colDim": im2col_dim, 
                    "Stride": stride
                }
    
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
                try: 
                    ker_dim = [2, layer.kernel_size, layer.kernel_size]
                    stride_dim = [2, layer.stride, layer.stride]
                except AttributeError:
                    ker_dim = [2, 8,8]
                    stride_dim = [2,1,1]

                input_dim = [3, input[0].shape[1],input[0].shape[2],input[0].shape[3]]
                output_dim = [3, output.shape[1], output.shape[2], output.shape[3]]
                
                parameters = {
                    "kerDim": ker_dim, 
                    "strideDim": stride_dim, 
                    "inputDim": input_dim,
                    "outputDim": output_dim
                }

            if isinstance(layer, nn.Linear):
                weight_dim = [2, layer.out_features, layer.in_features]
                biasPresent = 0 
                input_dim = [1, layer.in_features]
                output_dim = [1, layer.out_features]
                self.mvm += layer.in_features
                parameters = {
                    "weightDim": weight_dim, 
                    "biasPresent": biasPresent, 
                    "inputDim": input_dim, 
                    "outputDim": output_dim
                }
                
            ads_layers[name].set_parameters(parameters)

        return fn

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return 0

if args.model_arch == 'resnet18':
    model = models.resnet18(pretrained=True)
elif args.model_arch == 'resnet34':
    model = models.resnet34(pretrained=True)
elif args.model_arch == 'resnet50':
    model = models.resnet50(pretrained=True)

# 1- Instantiate our ModulePathTracer and use that to trace our model
tracer = ModulePathTracer()
traced_model = tracer.trace(model)

inputs = {}  # Register all inputs to the node 
outputs = {} # Register all outputs to the node 
for node in traced_model.nodes:
    module_qualname = tracer.node_to_originating_module.get(node)
    found = re.findall("Proxy\((.+?)\)", str(module_qualname))   # Extract origin list
    name = node.name

    inputs[name] = found
    for e in found: 
        if e not in outputs.keys():
            outputs[e] = [name]
        else: 
            outputs[e].append(name)

#print(inputs)
inputs['output'] = ['fc']
ads_layers = {}

ads_layers[0] = ADSlayer("IO", 0, "", "data_IO", 'IO', parameters= {"inputDim": [3,3,32,32]})
# generate intial file: 
i = 1
with open("test.ads", "w") as f:
    for k,v in inputs.items():
        if 'add' in k:
            type = 'Eltwise'
        elif 'conv' in k:
            type = "CONV"
        elif "leaky" in k or "relu" in k:
            type = "RELU"
        elif "batch_norm" in k or "bn" in k: 
            type = "BatchNorm"
        elif "pool" in k:
            type = "POOLING"
        elif "fc" in k:
            type = "IP"
        elif i == 1:
            type = "IO_IN"
        elif "cat" in  k:
            type ="Concat"
        elif "output" in k:
            type = "IO_OUT"
        elif "flatten" in k:
            continue
        else:
            type="unknown"


        if i == 1:
            ads_layers[k] = ADSlayer(k,i, ["data_IO"], "x", type)
        else:
            ads_layers[k] = ADSlayer(k,i, v, k, type)
        i +=1 
        
x = torch.randn(1,3,32,32) # initiate a random input to forward 
hook = HookLayerADSDefinition(model, inputs, outputs)
hook(x)
if hook.ignore == 0:
    for n,l in ads_layers.items():
        print(l)

sys.stdout.close()