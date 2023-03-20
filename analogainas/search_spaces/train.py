# TORCH IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR

# AIHWKIT IMPORTS
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType
from aihwkit.simulator.configs.utils import BoundManagementType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn import AnalogSequential
from aihwkit.optim import AnalogSGD

from analogainas.search_spaces.resnet_macro_architecture import Network
from analogainas.search_spaces.dataloaders.dataloader import load_cifar10

continue_analog = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_rpu_config(g_max=25,
                      tile_size=256,
                      dac_res=256,
                      adc_res=256,
                      noise_std=5.0):
    rpu_config = InferenceRPUConfig()

    rpu_config.clip.type = WeightClipType.FIXED_VALUE
    rpu_config.clip.fixed_value = 1.0
    rpu_config.modifier.pdrop = 0  # Drop connect.

    rpu_config.modifier.std_dev = noise_std

    rpu_config.modifier.rel_to_actual_wmax = True
    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 0.4
    rpu_config.mapping.weight_scaling_omega = True
    rpu_config.mapping.max_input_size = tile_size
    rpu_config.mapping.max_output_size = 255

    rpu_config.mapping.learn_out_scaling_alpha = True

    rpu_config.forward = PresetIOParameters()
    rpu_config.forward.inp_res = 1/dac_res  # 8-bit DAC discretization.
    rpu_config.forward.out_res = 1/adc_res  # 8-bit ADC discretization.
    rpu_config.forward.bound_management = BoundManagementType.NONE

    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config


def create_analog_optimizer(model, lr):
    optimizer = AnalogSGD(model.parameters(), lr=lr)
    optimizer.regroup_param_groups(model)

    return optimizer


def train(model, optimizer,  criterion, epoch, trainloader, testloader):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(trainloader),
              ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1),
                 100.*correct/total, correct, total))


def test(name, model, criterion, epoch, trainloader, testloader):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader),
                  'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss/(batch_idx+1),
                     100.*correct/total,
                     correct,
                     total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './{}.pth'.format(name))
        best_acc = acc


def digital_train(name, model, trainloader, testloader):
    model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    if torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    print(torch.cuda.is_available())
    lr = 0.05

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=400)

    for epoch in range(400):
        train(model, optimizer, criterion, epoch, trainloader, testloader)
        test(name, model, criterion, epoch, trainloader, testloader)
        scheduler.step()

        if epoch == 10:
            if best_acc < 20:
                continue_analog = False
                break


def analog_training(name, model, trainloader, testloader):
    name = name + "_analog"
    rpu_config = create_rpu_config()
    model_analog = convert_to_analog_mapped(model, rpu_config)
    # add this sequential to enable drift analog weights
    model_analog = AnalogSequential(model_analog)

    lr = 0.1
    epochs = 200
    model_analog.train()
    model_analog.to(device)

    optimizer = create_analog_optimizer(model_analog, lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(0, epochs):
        train(model_analog, optimizer, criterion,
              epoch, trainloader, testloader)
        test(name, model_analog, criterion,
             epoch, trainloader, testloader)
        model_analog.remap_analog_weights()
        scheduler.step()


def train_config(name, config):
    trainloader, testloader = load_cifar10(128)

    best_acc = 0     # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    net = Network(config)

    digital_train(name, net, trainloader, testloader)
    digital_acc = best_acc

    best_acc = 0.0
    start_epoch = 0
    if continue_analog:
        analog_training(name, net, trainloader, testloader)
        analog_acc = best_acc

    print(digital_acc)
    print(analog_acc)
