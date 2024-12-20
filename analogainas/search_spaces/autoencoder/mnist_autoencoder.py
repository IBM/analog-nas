import torch
import torch.nn as nn
import torch.nn.functional as F

from analogainas.search_spaces.autoencoder.autoencoder_architecture import AutoEncoder


class MnistAutoEncoder(nn.Module):
    def __init__(self, config, input_channels=1, input_size=(28,28)):
        super().__init__()
        self.model = AutoEncoder(config, input_channels, input_size)

    def forward(self, x):
        return self.model(x)