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

class HookLayerADSDefinition(nn.Module):
    """
    HookLayerADSDefinition: registers a hook in each layer to extract layer information
    such as kernel size, stride, padding... 
    and writes the corresponding ads definition. 
    """
    def __init__(self, model: nn.Module, inputs_, outputs_,down_sample):
        super().__init__()
        self.model = model
        #self.inputs = inputs_
        # Converting into list of tuple
        self.inputs  = [(k, v) for k, v in inputs_.items()]
        self.outputs = outputs_

        self.mvm = 0
        self.modules = {}
        self.inp = ["data"]
        self.not_down = down_sample
        self.bn2 = 0
        self.id = 1
        self.id_ = 1
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
            #print(self.id)
            print("Lid={},Ltype={}".format(self.id_, layer_to_ads[type(layer).__name__]))
            #print(output.shape)
            if isinstance(layer, nn.Conv2d):
                print("\tweightDim={}:{},{},{},{}".format(4, output.shape[1], input[0].shape[1], layer.kernel_size[0], layer.kernel_size[1]))
                biasPresent = 1 
                if not layer.bias: 
                    biasPresent = 0
                print("\tBiasPresent={}".format(biasPresent))
                print("\tinputDim={}:{},{}".format(2, input[0].shape[1], input[0].shape[2]*input[0].shape[3]))
                print("\toutputDim={}:{},{}".format(2, output.shape[1], output.shape[2]*output.shape[3]))
                im2colDim = compute_shape_col(input[0].shape, layer.kernel_size, layer.padding, layer.stride)
                print("\tim2colDim={}:{},{}".format(2, im2colDim[0], im2colDim[1]))
                self.mvm += im2colDim[1]
                #print(self.mvm)
                print("\tStride={}".format(layer.stride[0]))
                print("\trapaFactor=16")
                print("\twPrecision=8\n\tIPrecision=8\n\tOPrecision=8\n\tErrInPrecision=8\n\tErrOutPrecision=8")
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}".format(name))
                self.id += 1
                self.id_ +=1

            if isinstance(layer, nn.BatchNorm2d):
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}_".format(name))
                print("\tIPrecision=8\n\tOPrecision=8")

                print("Lid={},Ltype=Scale".format(self.id_+1))
                print("\tInp={}_".format(name))
                print("\tOut={}".format(name))
                print("\tIPrecision=8\n\tOPrecision=8")

                if "downsample_1" in name:
                    #print(self.id)
                    print("Lid={},Ltype=Eltwise\n\tInp={}\n\tInp={}\n\tOut={}\n\tIPrecision=8\n\tOPrecision=8".format(self.id_+2, self.inputs[self.id+1][1][1],  self.inputs[self.id+1][1][0], self.inputs[self.id+1][0]))
                    #print("Lid{},Ltype=RELU\n\tInp={}\n\tOut={}".format(self.id+2, self.inputs[self.id+2][1][0], self.inputs[self.id+2][0]))
                    self.id += 1
                    self.id_+= 1
                
                if "bn2" in name and self.id not in self.not_down:
                    print("Lid={},Ltype=Eltwise\n\tInp={}\n\tInp={}\n\tOut={}\n\tIPrecision=8\n\tOPrecision=8".format(self.id_+2, self.inputs[self.id+1][1][1],  self.inputs[self.id+1][1][0], self.inputs[self.id+1][0]))
                    #print("Lid{},Ltype=RELU\n\tInp={}\n\tOut={}".format(self.id+2, self.inputs[self.id+2][1][0], self.inputs[self.id+2][0]))
                    self.id += 1
                    self.id_+= 1

                if "bn2" in name:
                    self.bn2 += 1 
                    #print(self.bn2)

                self.id += 1
                self.id_+=2
            if isinstance(layer, nn.ReLU) or isinstance(layer, torch.nn.modules.activation.ReLU):
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}".format(self.inputs[self.id+1][1][0]))
                self.id += 1
                self.id_ += 1

            if isinstance(layer, nn.MaxPool2d):
                print("\tkerDim={}:{},{}".format(2, layer.kernel_size, layer.kernel_size))
                print("\tstrideDim={}:{},{}".format(2, layer.stride, layer.stride))
                print("\tinputDim={}:{},{},{}".format(3, input[0].shape[1],input[0].shape[2],input[0].shape[3]))
                print("\toutputDim={}:{},{},{}".format(3, output.shape[1], output.shape[2], output.shape[3]))
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}".format(name))
                self.id +=1
                self.id_ += 1

            if isinstance(layer, nn.AdaptiveAvgPool2d):
                print("\tkerDim={}:{},{}".format(2, input[0].shape[2],input[0].shape[3]))
                print("\tstrideDim={}:{},{}".format(2, 1, 1))
                print("\tinputDim={}:{},{},{}".format(3, input[0].shape[1],input[0].shape[2],input[0].shape[3]))
                print("\toutputDim={}:{},{},{}".format(3, output.shape[1], output.shape[2], output.shape[3]))
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}".format(name))
                self.id +=1
                self.id_ += 1

            if isinstance(layer, nn.Linear):
                print("\tweightDim={}:{},{}".format(2, layer.out_features, layer.in_features))
                biasPresent = 0 
                print("\tBiasPresent={}".format(biasPresent))
                print("\tinputDim={}:{}".format(1, layer.in_features))
                print("\toutputDim={}:{}".format(1, layer.out_features))
                print("\twPrecision=8\n\tIPrecision=8\n\tOPrecision=8\n\tErrInPrecision=8\n\tErrOutPrecision=8")
                print("\tInp={}".format(self.inputs[self.id][1][0]))
                print("\tOut={}".format(name))
                self.id += 1
                self.id_ += 1

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

# 2- Gather down_sampling layer indices.
down_sample = []
for i, (k, v) in enumerate(inputs.items()):
    if "downsample_1" in k:
        down_sample.append(i-2)


model.fc = nn.Linear(512, 2) # Only done for Visual Wake words task 
x = torch.randn(1,3,224,224) # initiate a random input to forward 
hook = HookLayerADSDefinition(model, inputs, outputs, down_sample)
hook(x)
sys.stdout.close()