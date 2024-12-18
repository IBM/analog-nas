import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from analogainas.evaluators.evaluation_metrics import negative_mse_metric
from analogainas.evaluators.realtime_rpu_evaluator import RealtimeRPUEvaluator
from analogainas.search_spaces.autoencoder.cifar_autoencoder import CifarAutoEncoder
from analogainas.search_algorithms.ea_optimized import EAOptimizer
from analogainas.search_algorithms.worker import Worker

from analogainas.search_spaces.config_spaces.rpu_config_space import RPUConfigSpace
from analogainas.search_spaces.dataloaders.autoencoder_structured_dataset import AutoEncoderStructuredDataset

import torch


from new_search_sample import evaluator
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

optimal_config={'embedding_dim': 256, 'encoder_convblock1_depth': 1, 'encoder_convblock1_kernel_size': 5, 'encoder_convblock1_filters': 64, 'encoder_convblock1_stride': 1, 'encoder_convblock2_depth': 1, 'encoder_convblock2_kernel_size': 3, 'encoder_convblock2_filters': 32, 'encoder_convblock2_stride': 1, 'encoder_convblock3_depth': 1, 'encoder_convblock3_kernel_size': 7, 'encoder_convblock3_filters': 16, 'encoder_convblock3_stride': 2}

autoencoder = CifarAutoEncoder(optimal_config)

criterion = nn.MSELoss()
optim = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 8

autoencoder.to(device)
autoencoder.train()

for epoch in range(epochs):
    for batch_idx, (images, _) in enumerate(dataloader):
        images = images.to(device)
        recon = sample_autoencoder(images)
        loss = criterion(recon, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}], Loss: {loss.item()}")

autoencoder = autoencoder.to(torch.device('cpu'))

CS = RPUConfigSpace()
print(CS.get_hyperparameters())

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

evaluator = RealtimeRPUEvaluator(model=autoencoder, metric_callback=negative_mse_metric, test_dataloader=test_dataloader, criterion=criterion, artifact_dir='CifarAutoEncoderTraining')

optimizer = EAOptimizer(evaluator, population_size=50, nb_iter=10, batched_evaluation=True)
#optimizer = EAOptimizer(evaluator, population_size=20, nb_iter=10, batched_evaluation=True)

NB_RUN = 1
worker = Worker(network_factory=CifarAutoEncoder, cs=CS, optimizer=optimizer, runs=NB_RUN)

print(worker.config_space)
worker.search()
