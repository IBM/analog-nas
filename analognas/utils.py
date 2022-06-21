import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from numpy.random import rand
from numpy.random import seed
from scipy.stats import kendalltau
from scipy.stats import spearmanr

def accuracy_mse(prediction, target, scale=100.0):
    prediction = prediction.detach() * scale
    target = (target) * scale
    return F.mse_loss(prediction, target)


ALPHA = 0.05 

'''
Compute the kendal correlation between two variables v1 & v2 
'''
def kendal_correlation(v1, v2):
    coef, p =  kendalltau(v1, v2)

    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0
    else:
        return coef 

'''
Compute the spearman correlation between two variables v1 & v2 
'''
def spearman_correlation(v1, v2):
    coef, p =  spearmanr(v1, v2)
    if p > ALPHA:
        print("Samples are uncorrelated (fail to reject H0)")
        return 0 
    else:
        return coef 

'''
Check if two variables contains ties. 
This can help us understand which one of the two rank correlation is more significant. 
Contains ties --> Spearman 
No ties --> Kendal
'''
def check_ties(v1, v2): 
    v1_set = set(v1) 
    v2_set = set(v2) 
    if len(v1_set.intersection(v2_set)) > 0: 
        return(True)  
    return(False)    


def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

'''
Compute the pairwise ranking error.
'''
def nb_rank_error(v1, v2):
    v1_sorted = sorted(range(len(v1)), key=lambda k: v1[k])
    v2_sorted = sorted(range(len(v2)), key=lambda k: v2[k])
    
    rank_errors = 0
    for i in range(len(v1)): 
        if v1_sorted[i] != v2_sorted[i]:
            rank_errors += 1
    return rank_errors

##################################
# Architecture Metrics 
'''
Compute the number of parameters of model.
'''
def get_nb_params(model):
    return sum(p.numel() for p in model.parameters())

'''
Compute the depth of the model.
'''
def get_nb_convs(config):
    m = config["M"]
    nb_conv = 0
    for i in range(1,m+1):
        if config["convblock%d" % i] == 1:
            nb_conv += config["R%d" % i]*2*config["B%d" % i]
        if config["convblock%d" % i] == 2:
            nb_conv += config["R%d" % i]*3*config["B%d" % i]
    return nb_conv

##################################