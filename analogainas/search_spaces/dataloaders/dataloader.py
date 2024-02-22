import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from analogainas.search_spaces.dataloaders.cutout import Cutout

import importlib.util
pyvww = importlib.util.find_spec("pyvww")
found = pyvww is not None

def load_cifar10(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        Cutout(1, length=8)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    testloader = DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def load_vww(batch_size, path, annot_path):
    transform = transforms.Compose([
        transforms.CenterCrop(100),
        transforms.ToTensor()
    ])

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
                    root=path, annFile=annot_path, transform=transform)
    valid_dataset = pyvww.pytorch.VisualWakeWordsClassification(
                    root=path, annFile=annot_path, transform=transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=1)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1)

    return train_loader, valid_loader
