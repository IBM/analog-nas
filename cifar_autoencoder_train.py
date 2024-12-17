import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from analogainas.search_spaces.autoencoder.cifar_autoencoder import CifarAutoEncoder
from analogainas.search_spaces.config_space import ConfigSpace
from analogainas.search_spaces.autoencoder.autoencoder_config_space import AutoEncoderConfigSpace
from analogainas.search_spaces.autoencoder.autoencoder_architecture import AutoEncoder
from analogainas.search_spaces.autoencoder.mnist_autoencoder import MnistAutoEncoder
from analogainas.search_spaces.autoencoder.mnist_autoencoder import MnistAutoEncoder
from analogainas.evaluators.realtime_training_evaluator import RealtimeTrainingEvaluator
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker
from analogainas.search_spaces.dataloaders.autoencoder_structured_dataset import AutoEncoderStructuredDataset

from analogainas.analog_helpers.analog_helpers import create_rpu_config, create_analog_optimizer
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType
from aihwkit.simulator.configs.utils import BoundManagementType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn import AnalogSequential
from aihwkit.optim import AnalogSGD


CS = AutoEncoderConfigSpace()

print(CS.get_hyperparameters())

print(CS.compute_cs_size())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

train_cifar_dataset = AutoEncoderStructuredDataset(
    torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
)

train_dataloader = DataLoader(train_cifar_dataset, batch_size=8, shuffle=True)

test_cifar_dataset = AutoEncoderStructuredDataset(
    torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
)

test_dataloader = DataLoader(test_cifar_dataset, batch_size=64, shuffle=True)

criterion = nn.MSELoss()
evaluator = RealtimeTrainingEvaluator(model_factory=MnistAutoEncoder, train_dataloader=train_dataloader, val_dataloader=test_dataloader, test_dataloader=test_dataloader, criterion=criterion, epochs=10, artifact_dir='LongAutoEncoderTraining')

optimizer = EAOptimizer(evaluator, population_size=50, nb_iter=10, batched_evaluation=True)
#optimizer = EAOptimizer(evaluator, population_size=20, nb_iter=10, batched_evaluation=True)

NB_RUN = 1
worker = Worker(network_factory=CifarAutoEncoder, cs=CS, optimizer=optimizer, runs=NB_RUN)

print(worker.config_space)
worker.search()
