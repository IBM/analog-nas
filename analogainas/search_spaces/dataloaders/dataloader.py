import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from analogainas.search_spaces.dataloaders.cutout import Cutout

import importlib.util

pyvww = importlib.util.find_spec("pyvww")
found = pyvww is not None

# Custom Dataset for Nuclei Segmentation
from analogainas.search_spaces.dataloaders.Nuclei_Dataset import Dataset as Dataset

# Digital Training imports
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

# MONAI Imports for compatibility
from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    RandRotate90,
    RandFlip,
    OneOf,
    RandAdjustContrast,
    RandGaussianNoise,  # MONAI does not directly support RandomBrightness, using Gaussian noise as an alternative
    RandGaussianSharpen,  # MONAI does not have a direct match for RandomContrast, using Gaussian sharpen as an alternative
    RandShiftIntensity,  # Alternative for adjusting brightness
    Resize,
    NormalizeIntensity,
)


# def load_spleen():
#     resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
#     md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

#     root_dir = os.getcwd()
#     compressed_file = os.path.join(root_dir, "Task09_Spleen.tar")
#     data_dir = os.path.join(root_dir, "Task09_Spleen")
#     if not os.path.exists(data_dir):
#         download_and_extract(resource, compressed_file, root_dir, md5)

#     train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
#     train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
#     data_dicts = [
#         {"image": image_name, "label": label_name}
#         for image_name, label_name in zip(train_images, train_labels)
#     ]
#     train_files, val_files = data_dicts[:-9], data_dicts[-9:]
#     transform_train = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             ScaleIntensityRanged(
#                 keys=["image"],
#                 a_min=-57,
#                 a_max=164,
#                 b_min=0.0,
#                 b_max=1.0,
#                 clip=True,
#             ),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#             RandCropByPosNegLabeld(
#                 keys=["image", "label"],
#                 label_key="label",
#                 spatial_size=(96, 96, 96),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="image",
#                 image_threshold=0,
#             ),
#         ]
#     )
#     transform_test = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             ScaleIntensityRanged(
#                 keys=["image"],
#                 a_min=-57,
#                 a_max=164,
#                 b_min=0.0,
#                 b_max=1.0,
#                 clip=True,
#             ),
#             CropForegroundd(keys=["image", "label"], source_key="image"),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             Spacingd(
#                 keys=["image", "label"],
#                 pixdim=(1.5, 1.5, 2.0),
#                 mode=("bilinear", "nearest"),
#             ),
#         ]
#     )

#     train_ds = CacheDataset(
#         data=train_files, transform=transform_train, cache_rate=1.0, num_workers=4
#     )
#     trainloader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)

#     val_ds = CacheDataset(
#         data=val_files, transform=transform_test, cache_rate=1.0, num_workers=4
#     )
#     testloader = DataLoader(val_ds, batch_size=1, num_workers=4)

#     return trainloader, testloader


def load_cifar10(batch_size):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            Cutout(1, length=8),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_vww(batch_size, path, annot_path):
    transform = transforms.Compose([transforms.CenterCrop(100), transforms.ToTensor()])

    train_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=path, annFile=annot_path, transform=transform
    )
    valid_dataset = pyvww.pytorch.VisualWakeWordsClassification(
        root=path, annFile=annot_path, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=1
    )

    return train_loader, valid_loader


# Loader Configs
config = {
    "spatial_dims": 2,  # it's 2 for 2D images; would be 3 for 3D images
    "in_channels": 3,  # e.g., 3 for RGB input images
    "out_channels": 1,  # e.g., 1 for binary segmentation tasks
    "channels": (16, 32, 64, 128, 256),  # Number of channels in the inner layers
    "strides": (2, 2, 2, 2),
    "name": None,
    "epochs": 200,
    "batch_size": 64,
    "arch": "UNet",
    "deep_supervision": False,
    "input_channels": 3,
    "num_classes": 1,
    "input_w": 96,
    "input_h": 96,
    "loss": "BCEDiceLoss",
    "dataset": "dsb2018_96",
    "img_ext": ".png",
    "mask_ext": ".png",
    "optimizer": "SGD",
    "lr": 1e-3,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "nesterov": False,
    "scheduler": "CosineAnnealingLR",
    "min_lr": 1e-5,
    "factor": 0.1,
    "patience": 2,
    "milestones": "1,2",
    "gamma": 2 / 3,
    "early_stopping": -1,
    "num_workers": 4,
}


def load_nuclei_dataset():
    # Data loading code
    img_ids = glob(os.path.join("../../inputs", "dsb2018_96", "images", "*" + ".png"))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(
        img_ids, test_size=0.2, random_state=41
    )

    train_transform = Compose(
        [
            RandRotate90(
                prob=0.5, spatial_axes=(0, 1)
            ),  # Randomly rotates the images by 90 degrees with a default probability of 0.5
            RandFlip(
                prob=0.5, spatial_axis=0
            ),  # Randomly flips the image along a given axis
            OneOf(
                [  # Applies one of the transforms
                    RandShiftIntensity(
                        offsets=0.1, prob=1.0
                    ),  # Randomly shifts intensity for brightness adjustment
                    RandAdjustContrast(prob=1.0),  # Randomly changes contrast
                    RandGaussianSharpen(prob=1.0),  # Randomly sharpens the image
                ]
            ),
            Resize(
                spatial_size=(config["input_h"], config["input_w"])
            ),  # Resize images to a specified size
            NormalizeIntensity(
                nonzero=True, channel_wise=True
            ),  # Normalize pixel values with channel-wise option
        ]
    )

    val_transform = Compose(
        [
            Resize(
                spatial_size=(config["input_h"], config["input_w"])
            ),  # Resize images to a specified size
            NormalizeIntensity(
                nonzero=True, channel_wise=True
            ),  # Normalize pixel values with channel-wise option
        ]
    )

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join("inputs", config["dataset"], "images"),
        mask_dir=os.path.join("inputs", config["dataset"], "masks"),
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        transform=train_transform,
    )

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join("inputs", config["dataset"], "images"),
        mask_dir=os.path.join("inputs", config["dataset"], "masks"),
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )
    return train_loader, val_loader
