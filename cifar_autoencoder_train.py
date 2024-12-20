import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from analogainas.search_spaces.autoencoder.cifar_autoencoder import CifarAutoEncoder
from analogainas.search_spaces.autoencoder.autoencoder_config_space import AutoEncoderConfigSpace
from analogainas.evaluators.realtime_training_evaluator import RealtimeTrainingEvaluator
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker
from analogainas.search_spaces.dataloaders.autoencoder_structured_dataset import AutoEncoderStructuredDataset



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
evaluator = RealtimeTrainingEvaluator(model_factory=CifarAutoEncoder, train_dataloader=train_dataloader, val_dataloader=test_dataloader, test_dataloader=test_dataloader, criterion=criterion, epochs=13, artifact_dir='CifarAutoEncoderTraining')

optimizer = EAOptimizer(evaluator, population_size=50, nb_iter=10, batched_evaluation=True)

NB_RUN = 1
worker = Worker(network_factory=CifarAutoEncoder, cs=CS, optimizer=optimizer, runs=NB_RUN)

print(worker.config_space)
worker.search()
