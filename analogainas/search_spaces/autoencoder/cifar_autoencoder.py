import torch
import torch.nn as nn
import torch.nn.functional as F

from analogainas.search_spaces.autoencoder.autoencoder_architecture import AutoEncoder

class CifarAutoEncoder(nn.Module):
    def __init__(self, config, input_channels=3, input_size=(32, 32)):
        super().__init__()
        self.model = AutoEncoder(config, input_channels, input_size)

    def forward(self, x):
        return self.model(x)