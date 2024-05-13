# TORCH IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd

# AIHWKIT IMPORTS
# from aihwkit.simulator.configs import InferenceRPUConfig
# from aihwkit.simulator.configs.utils import WeightClipType
# from aihwkit.simulator.configs.utils import BoundManagementType
# from aihwkit.simulator.presets.utils import PresetIOParameters
# from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
# from aihwkit.inference.compensation.drift import GlobalDriftCompensation
# from aihwkit.nn.conversion import convert_to_analog_mapped
# from aihwkit.nn import AnalogSequential
# from aihwkit.optim import AnalogSGD

from analogainas.search_spaces.resnet_macro_architecture import Network
from analogainas.search_spaces.dataloaders.dataloader import load_cifar10
from analogainas.search_spaces.dataloaders.dataloader import load_nuclei_dataset

from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
)

from tqdm import tqdm
from collections import OrderedDict

continue_analog = True
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def create_rpu_config(g_max=25, tile_size=256, dac_res=256, adc_res=256, noise_std=5.0):
#     rpu_config = InferenceRPUConfig()

#     rpu_config.clip.type = WeightClipType.FIXED_VALUE
#     rpu_config.clip.fixed_value = 1.0
#     rpu_config.modifier.pdrop = 0  # Drop connect.

#     rpu_config.modifier.std_dev = noise_std

#     rpu_config.modifier.rel_to_actual_wmax = True
#     rpu_config.mapping.digital_bias = True
#     rpu_config.mapping.weight_scaling_omega = 0.4
#     rpu_config.mapping.weight_scaling_omega = True
#     rpu_config.mapping.max_input_size = tile_size
#     rpu_config.mapping.max_output_size = 255

#     rpu_config.mapping.learn_out_scaling_alpha = True

#     rpu_config.forward = PresetIOParameters()
#     rpu_config.forward.inp_res = 1 / dac_res  # 8-bit DAC discretization.
#     rpu_config.forward.out_res = 1 / adc_res  # 8-bit ADC discretization.
#     rpu_config.forward.bound_management = BoundManagementType.NONE

#     # Inference noise model.
#     rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)

#     # drift compensation
#     rpu_config.drift_compensation = GlobalDriftCompensation()
#     return rpu_config


# def create_analog_optimizer(model, lr):
#     optimizer = AnalogSGD(model.parameters(), lr=lr)
#     optimizer.regroup_param_groups(model)

#     return optimizer


# IOU Score and DICE Coefficients
import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


# Loss Functions for Segmentation
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = (2.0 * intersection.sum(1) + smooth) / (
            input.sum(1) + target.sum(1) + smooth
        )
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


def train(train_loader, model, criterion, optimizer):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss = criterion(output, target)
        iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters["loss"].update(loss.item(), input.size(0))
        avg_meters["iou"].update(iou, input.size(0))

        postfix = OrderedDict(
            [
                ("loss", avg_meters["loss"].avg),
                ("iou", avg_meters["iou"].avg),
            ]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
    )


def test(val_loader, model, criterion):
    global best_acc
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

            avg_meters["loss"].update(loss.item(), input.size(0))
            avg_meters["iou"].update(iou, input.size(0))

            postfix = OrderedDict(
                [
                    ("loss", avg_meters["loss"].avg),
                    ("iou", avg_meters["iou"].avg),
                ]
            )
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
    )


def digital_train(model, trainloader, testloader):
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = model.to(device)
        # cudnn.benchmark = True

    print(torch.cuda.is_available())

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=400)
    # dice_metric = DiceMetric(include_background=False, reduction="mean")
    # metric_values = []
    # epoch_loss_values = []
    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
        ]
    )

    best_iou = 0
    trigger = 0
    for epoch in range(100):
        train_log = train(trainloader, model, criterion, optimizer)
        test_log = test(
            testloader,
            model,
            criterion,
        )
        scheduler.step()
        log["epoch"].append(epoch)
        log["lr"].append(1e-3)
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(test_log["loss"])
        log["val_iou"].append(test_log["iou"])

        pd.DataFrame(log).to_csv("models/NasSegNet/log.csv", index=False)

        trigger += 1

        if test_log["iou"] > best_iou:
            torch.save(model.state_dict(), "models/NasSegNet/model.pth")
            best_iou = test_log["iou"]
            print("=> saved best model")
            trigger = 0

        # if epoch == 10:
        #     if best_acc < 20:
        #         continue_analog = False
        #         break


# def analog_training(name, model, trainloader, testloader):
#     name = name + "_analog"
#     rpu_config = create_rpu_config()
#     model_analog = convert_to_analog_mapped(model, rpu_config)
#     # add this sequential to enable drift analog weights
#     model_analog = AnalogSequential(model_analog)

#     lr = 0.1
#     epochs = 200
#     model_analog.train()
#     model_analog.to(device)

#     optimizer = create_analog_optimizer(model_analog, lr)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
#     criterion = nn.CrossEntropyLoss().to(device)

#     for epoch in range(0, epochs):
#         train(model_analog, optimizer, criterion, epoch, trainloader, testloader)
#         test(name, model_analog, criterion, epoch, trainloader, testloader)
#         model_analog.remap_analog_weights()
#         scheduler.step()


def train_config_unet(config):
    trainloader, testloader = load_nuclei_dataset()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    net = Network(config)

    digital_train(net, trainloader, testloader)
    digital_acc = best_acc

    # best_acc = 0.0
    # start_epoch = 0
    # if continue_analog:
    #     analog_training(name, net, trainloader, testloader)
    #     analog_acc = best_acc

    print(digital_acc)
    # print(analog_acc)


# def train_config(name, config):
#     trainloader, testloader = load_cifar10(128)

#     best_acc = 0  # best test accuracy
#     start_epoch = 0  # start from epoch 0 or last checkpoint epoch

#     net = Network(config)

#     digital_train(name, net, trainloader, testloader)
#     digital_acc = best_acc

#     best_acc = 0.0
#     start_epoch = 0
#     if continue_analog:
#         analog_training(name, net, trainloader, testloader)
#         analog_acc = best_acc

#     print(digital_acc)
#     print(analog_acc)
