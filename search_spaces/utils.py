import os
import math
import torch
import torch.nn as nn
import importlib


def load_model(config, checkpoint):
    module = importlib.import_module('model.' + config.arch)
    Network = getattr(module, 'Network')(config)

    if checkpoint is not None:
        model_ckpt = checkpoint['checkpoint']
        model_path = os.path.join(config.resume, model_ckpt)
        assert(os.path.exists(model_path), 'Saved model not found: ' + model_path)
        print('=> Resuming model from ' + model_path)
        model_stats = torch.load(model_path)
        Network.load_state_dict(model_stats['state'])
    else:
        print('=> Creating model from file: model/' + config.arch + '.py')
    
    return Network


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        module.weight.data.normal_(0, math.sqrt(2. / n))
        #nn.init.kaiming_normal(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()