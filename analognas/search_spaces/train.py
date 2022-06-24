# TORCH IMPORTS 
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from cutout import * 
# AIHWKIT IMPORTS 
from aihwkit.simulator.configs import InferenceRPUConfig              # Applies the noises in the forward during hw-training 
from aihwkit.simulator.configs.utils import WeightClipType            # Can be: FIXED_VALUE, LAYER_GAUSSIAN (std)
                                                                      # if layer_gaussian: Min(std, fixed_value)
from aihwkit.simulator.configs.utils import WeightModifierType        # weight noise application function (additive gaussian, discretize...)
from aihwkit.simulator.configs.utils import WeightNoiseType           # applied to the output y = xw + noise
from aihwkit.simulator.configs.utils import BoundManagementType       # adjust the output 
from aihwkit.simulator.presets.utils import PresetIOParameters        # Defines the hardware configuration used in the forward and backward passes. 
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel             # Noise model that was fitted and characterized on real PCM devices
from aihwkit.inference.compensation.drift import GlobalDriftCompensation # applies a constant factor to compensate for the drift.

from aihwkit.nn.conversion import convert_to_analog_mapped                  # converts Pytorch model to analog,  _mapped to use multiple tiles per layer.
from aihwkit.nn import AnalogSequential                             
from aihwkit.optim import AnalogSGD

from analognas.search_spaces.resnet_macro_architecture import Network 

continue_analog = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CREATE RPU CONFIGURATION 
# (this rpu config will be applied to all layers of the model)
def create_rpu_config(g_max = 25, tile_size=256, dac_res = 256, adc_res = 256, noise_std=5.0):
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
    rpu_config.forward.out_res = 1/adc_res # 8-bit ADC discretization.
    rpu_config.forward.bound_management = BoundManagementType.NONE
    
    # Inference noise model.
    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)

    # drift compensation
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

rpu_config = create_rpu_config()
#######################################################################################
# LOAD DATASET 
#######################################################################################
def create_analog_optimizer(model, lr):
    optimizer = AnalogSGD(model.parameters(), lr=lr) 
    optimizer.regroup_param_groups(model)

    return optimizer
#######################################################################################
# LOAD DATASET 
#######################################################################################
def load_cifar10(batch_size): 
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout(1, length=8)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

classes = ('plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck')
trainloader, testloader = load_cifar10(128)
######################################################################################
# TRAIN
######################################################################################
def train(model, optimizer,  criterion, epoch):
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

        print(batch_idx, len(trainloader), ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
#######################################################################################
# Test 
#######################################################################################
def test(name, model, criterion, epoch):
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

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
##################################################################################
# Digital training 
##################################################################################
def digital_train(name, model):
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)

    for epoch in range(400):
        train(model, optimizer, criterion, epoch)
        test(name, model, criterion, epoch)
        scheduler.step()

        if epoch == 10:
            if best_acc < 20:
                continue_analog = False
                break

####################################################################################
# Analog training 
####################################################################################
def analog_training(name, model):
    name = name + "_analog"
    rpu_config = create_rpu_config()
    model_analog = convert_to_analog_mapped(model, rpu_config)
    model_analog = AnalogSequential(model_analog) # add this sequential to enable drift analog weights 

    input_size = (1, 3, 32, 32)
    lr = 0.1
    epochs = 200
    model_analog.train()
    model_analog.to(device)

    optimizer = create_analog_optimizer(model_analog, lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(0, epochs):
        train(model_analog, optimizer, criterion, epoch)
        test(name, model_analog, criterion, epoch)
        model_analog.remap_analog_weights() # remapping the weights 
        scheduler.step()
##################################################################################
def train(name, config):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    net = Network(config)
    
    digital_train(name, net)
    digital_acc = best_acc

    best_acc = 0.0
    start_epoch = 0 
    if continue_analog:
        analog_training(name, net)
        analog_acc = best_acc 

    print(digital_acc)
    print(analog_acc)






    
    


