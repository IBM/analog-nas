# Example of training script



EPOCHS = 200
QAT_EPOCHS = 20
EVALS = 25

# Standard Library imports
import os
import sys
import time
import random
import json
import copy
import fcntl
from pathlib import Path
import warnings

# Third-party Libraries
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Box SDK
from boxsdk import OAuth2, Client

# NASLib imports
import naslib
from naslib.search_spaces import NasBench201SearchSpace
from naslib.utils import get_dataset_api
from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str

# AIHWKit imports
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import StandardHWATrainingPreset
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightModifierType, BoundManagementType, WeightClipType, NoiseManagementType, WeightRemapType
)
from aihwkit.simulator.presets.utils import IOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.optim import AnalogSGD
from aihwkit.utils.analog_info import analog_summary

# PyTorch general imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.amp import GradScaler, autocast

# Torchvision
from torchvision import datasets, transforms

# PyTorch AO Quantization
import torch.ao.quantization
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import (
    get_default_qconfig_mapping,
    get_default_qat_qconfig_mapping,
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.qconfig import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

# PyTorch Quantization Prototype (torchao)
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
from torchao.quantization.quant_api import (
    quantize_,
    int8_dynamic_activation_int8_weight,
)

# Miscellaneous imports
from torchinfo import summary


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

warnings.filterwarnings("ignore")

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            image = image.to("cpu")
            model(image)

def save_results_safely(results_dict, architecture, rank):
    """
    Safely save results to CSV file with proper file locking for multi-process access
    """
    results_file = 'analog_nasbench201_off.csv'
    lock_file = 'analog_nasbench201_off.lock'
    
    # Create a temporary file for this process
    temp_file = f'temp_results_{rank}_{architecture}.json'
    
    # Save the results dictionary to a temporary JSON file
    with open(temp_file, 'w') as f:
        json.dump(results_dict, f)
    
    # Acquire lock for CSV operations
    with open(lock_file, 'w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Read existing CSV if it exists
            if os.path.exists(results_file):
                df = pd.read_csv(results_file)
            else:
                df = pd.DataFrame(columns=[
                    'architecture', 'baseline_accuracy', 'ptq_accuracy','qat_accuracy', 'analog_accuracy', 
                    'finetuned_accuracy', 'baseline_drift_60', 'baseline_drift_3600', 
                    'baseline_drift_86400', 'baseline_drift_2592000', 'finetuned_drift_60',
                    'finetuned_drift_3600', 'finetuned_drift_86400', 'finetuned_drift_2592000'
                ])
            
            # Load the temporary results and append
            with open(temp_file, 'r') as f:
                new_row = json.load(f)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Save the updated DataFrame
            df.to_csv(results_file, index=False)
            
        finally:
            # Release the lock
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

results_df = pd.DataFrame(columns=['architecture', 'baseline_accuracy', 'qat_accuracy', 'analog_accuracy', 'finetuned_accuracy',
    'baseline_drift_60', #1 minute
    'baseline_drift_3600', #1 hour
    'baseline_drift_86400', # 24 hours
    'baseline_drift_2592000', #30 days

    'finetuned_drift_60',
    'finetuned_drift_3600',
    'finetuned_drift_86400',
    'finetuned_drift_2592000'])

def print_size_of_model(model, architecture):
    torch.save(model.state_dict(), f"temp_{architecture}.p")
    print('Size (MB)\t', os.path.getsize(f"temp_{architecture}.p")/1e6)
    os.remove(f"temp_{architecture}.p")


def calculate_model_size(model):
    """
    Calculate the size of a PyTorch model in bytes.
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    return total_size


def gen_rpu_config():
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.std_dev = 0.06
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0

    rpu_config.forward = IOParameters()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1 / (2**8 - 2)
    rpu_config.forward.out_bound = 10
    rpu_config.forward.out_res = 1 / (2**8 - 2)
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.decay = 0.01
    rpu_config.pre_post.input_range.init_from_data = 50
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.input_min_percentage = 0.995
    rpu_config.pre_post.input_range.manage_output_clipping = False

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config

def create_model(architecture):
    model = NasBench201SearchSpace(n_classes=10)
    model.set_spec(architecture)
    model.parse()
    return model

def evaluate_model(model, test_loader, device, rank, architecture, runs=EVALS):
    model.eval()
    accuracies = []
    for run in range(runs):
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        # print(f"Rank {rank}: Test Accuracy (run {run+1}): {accuracy:.2f}%")
    avg_accuracy = sum(accuracies) / len(accuracies)

    print(f"Evaluation... {architecture}: Average Test Accuracy over {runs} runs: {avg_accuracy:.2f}%")

    return avg_accuracy

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def train(rank, world_size, architecture, epochs=EPOCHS):

    folder_name = "results"

    os.makedirs(folder_name, exist_ok=True)

    log_file = f"./{folder_name}/out_{architecture}.log"
    sys.stdout = open(log_file, "w")
    sys.stderr = open(log_file, "w")

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)


    node_name = os.environ.get("SLURM_NODELIST", "Node name not found")
    print(f"This script is running on node: {node_name}")

    seed = 42 + rank  # Different seed per process but reproducible
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    start_time = time.time()

    # DATASET CIFAR10 ------------------------------------------------------------------------------------------------------
    print(f"Begin dataset loading")
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True )
    print(f"Data loader setup complete")
    #-----------------------------------------------------------------------------------------------------------------------

    print(f"Training arch: {architecture} on device {device}")
    model = create_model(architecture).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)


    scaler = GradScaler()
    #training---------------------------------------------------------------------------------------------------------
    print(f"Begin training\n")
    model.train()
    accuracies_epochs = []
    losses_epochs = []
    validation_epochs = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler.step()

        train_accuracy = 100 * correct / total
        avg_loss = total_loss / total
        # validation_accuracy = evaluate_model(model, test_loader,device,rank,architecture,runs=1)

        print(f"{architecture}`\t Epoch [{epoch+1}/{epochs}]\t Loss: {avg_loss:.4f}\t Training Accuracy: {train_accuracy:.2f}%")
        accuracies_epochs.append(train_accuracy)
        losses_epochs.append(avg_loss)
        # validation_epochs.append(validation_accuracy)
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    print(f"\nEnd training\n")

    training_end_time = time.time()
    print(f"train time : {training_end_time - start_time}")
    
    #save_model----------------------------------------------------------------------
    model_save_path = f"./models_saved/model_{architecture}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    #-----------------------------------------------------------------------------------


    #upload the saved model to Box
    #upload_to_box(config, model_save_path)


    baseline_accuracy = evaluate_model(model, test_loader, device, rank, architecture)

    eval_end_time = time.time()
    print(f"evaluation time : {eval_end_time - training_end_time}\n")
    
    #post training quant
    print("Post training quantization...")
    model_copy = copy.deepcopy(model)
    model_copy.to("cpu")
    model_copy.eval()
    example_inputs = (next(iter(test_loader))[0])

    qconfig = get_default_qconfig("x86")
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    print("prepare model for ptq...")
    prepared_ptq_model = prepare_fx(model_copy, qconfig_mapping, example_inputs)
    print("premare model for ptq... DONE!\n")

    print("calibration...")
    calibrate(prepared_ptq_model, test_loader)
    print("calibration... DONE!\n")

    print("convert ptq...")
    ptq_model = convert_fx(prepared_ptq_model)
    print("convert ptq... DONE!\n")

    ptq_end_time = time.time()
    print(f"ptq time : {ptq_end_time - eval_end_time}\n")



    ptq_model.to("cpu")
    ptq_accuracy = evaluate_model(ptq_model,test_loader,device="cpu", rank=rank, architecture=architecture)



    debut_qat_time = time.time()
    #quantization QAT----------------------------------------------------------------------------------------
    print("Quantization-Aware Training...")
    
    model_to_quantize = copy.deepcopy(model)
    qconfig_mapping = get_default_qat_qconfig_mapping("x86")
    model_to_quantize.to('cpu')
    prepared_qat_model = quantize_fx.prepare_qat_fx(model_to_quantize, qconfig_mapping, example_inputs)

    print("Training QAT Model...")
    prepared_qat_model.train()
    qat_optimizer = optim.Adam(prepared_qat_model.parameters(), 
                             lr=0.001,
                             weight_decay=1e-4)
    qat_scheduler = optim.lr_scheduler.ReduceLROnPlateau(qat_optimizer, 
                                                    mode='max',
                                                    factor=0.1,
                                                    patience=3)
    criterion = torch.nn.CrossEntropyLoss()
    prepared_qat_model.to(device)
    
    for epoch in range(QAT_EPOCHS):
        # Training phase
        prepared_qat_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            qat_optimizer.zero_grad()
            
            outputs = prepared_qat_model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(prepared_qat_model.parameters(), max_norm=1.0)
            
            qat_optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100 * correct / total
        
        # Validation phase
        prepared_qat_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = prepared_qat_model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_acc = 100 * val_correct / val_total
        
        # Update scheduler based on validation accuracy
        qat_scheduler.step(val_acc)
        
        print(f'QAT Epoch: {epoch+1}/{QAT_EPOCHS}')
        print(f'QAT Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

    prepared_qat_model.to('cpu')
    
    qat_model = quantize_fx.convert_fx(prepared_qat_model)


    fin_qat_time = time.time()
    print(f"\nqat time : {fin_qat_time - debut_qat_time}\n")


    print("EVAL QAT Model...")
    qat_accuracy = evaluate_model(qat_model, test_loader, "cpu", rank, architecture)


    #convert to analog
    print("ANALOG CONVERSION\n")
    analog_model = convert_to_analog(model, gen_rpu_config())


    print("EVAL ANALOG Model...")
    #evaluate analog model
    analog_accuracy = evaluate_model(analog_model, test_loader, device, rank, architecture)

    #fine tuning
    criterion_analog = nn.CrossEntropyLoss()
    optimizer_analog = AnalogSGD(analog_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler_analog = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    scaler = GradScaler()

    debut_finetuning_time = time.time()

    analog_model.train()

    accuracies_f_epochs = []
    losses_f_epochs = []
    validation_f_epochs = []

    print("Fintuning analog model...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_analog.zero_grad()
            outputs = analog_model(inputs)

            loss = criterion_analog(outputs, targets)
            loss.backward()
            optimizer_analog.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        scheduler_analog.step()

        train_accuracy = 100 * correct / total
        avg_loss = total_loss / total
        # validation_accuracy = evaluate_model(analog_model, test_loader, device, rank, architecture, runs=1)

        print(f"Finetuning : Arch {architecture}\t Epoch [{epoch+1}/{epochs}]\t Loss: {avg_loss:.4f}\t Training Accuracy: {train_accuracy:.2f}%")
        accuracies_f_epochs.append(train_accuracy)
        losses_f_epochs.append(avg_loss)
        # validation_f_epochs.append(validation_accuracy)

    fin_finetuning_time = time.time()
    print(f"\nfinetuning time : {fin_finetuning_time - debut_finetuning_time}\n")

    training_data = {
        "architecture" : str(architecture),
        "epochs_training" : [
            {"epoch": epoch+1, "accuracy":accuracies_epochs[epoch], "loss":losses_epochs[epoch]}
            for epoch in range(len(accuracies_epochs))
        ],
        "epochs_finetuning" : [
            {"epoch": epoch+1, "accuracy":accuracies_f_epochs[epoch], "loss":losses_f_epochs[epoch]}
            for epoch in range(len(accuracies_f_epochs))
        ]
    }
    json_name = f"./{folder_name}/training_results_{architecture}.json"
    with open(json_name, "w") as json_file:
        json.dump(training_data, json_file, indent=4)

    print(f"Training results saved to {json_name}\n")


    #evaluate finetuning
    finetuned_accuracy = evaluate_model(analog_model, test_loader, device, rank, architecture)  


    print("Summary:")
    print(f"baseline accuracy\t {baseline_accuracy}")
    print_size_of_model(model, architecture)
    print(f"ptq accuracy\t {ptq_accuracy}")
    print_size_of_model(ptq_model, architecture)
    print(f"qat accuracy\t {qat_accuracy}")
    print_size_of_model(qat_model, architecture)
    print(f"analog accuracy\t {analog_accuracy}")
    print(f"finetuned accuracy\t {finetuned_accuracy}")
    print_size_of_model(analog_model, architecture)

    
    debut_temporal_time = time.time()
    #drift
    print("\n\n--- Temporal Drift ---")
    converted_model = convert_to_analog(model, gen_rpu_config())
    converted_model = converted_model.eval()
    analog_model = analog_model.eval()


    n_rep = 25
                # 1 minute, 1 hour, 1 day, 30 days
    t_inferences = [60., 3600., 86400., 2592000.]
    drifted_test_accs_finetuned = torch.zeros(size=(len(t_inferences),n_rep))
    drifted_test_accs_baseline = torch.zeros(size=(len(t_inferences),n_rep))

    for i,t in enumerate(t_inferences):
        print(t)
        for j in range(n_rep):
            converted_model.drift_analog_weights(t)
            drifted_test_accs_baseline[i,j] = evaluate_model(converted_model, test_loader, device, rank, architecture, runs=1)
            analog_model.drift_analog_weights(t)
            drifted_test_accs_finetuned[i,j] = evaluate_model(analog_model, test_loader, device, rank, architecture, runs=1)
    

    print(f"{'Inference Time (s)':<20}{'HW-aware Accuracy':<25}{'Baseline Accuracy':<25}")
    print("-" * 70)
    for i, t in enumerate(t_inferences):
        hw_aware_mean = drifted_test_accs_finetuned[i].mean().item()
        hw_aware_std = drifted_test_accs_finetuned[i].std().item()
        baseline_mean = drifted_test_accs_baseline[i].mean().item()
        baseline_std = drifted_test_accs_baseline[i].std().item()
        print(f"{t:<20.1f}{hw_aware_mean:.4f} ± {hw_aware_std:.4f}    {baseline_mean:.4f} ± {baseline_std:.4f}")

    fin_temporal_time = time.time()
    print(f"\ntemporal drift time : {fin_temporal_time - debut_temporal_time}\n\n")

    print("Summary")
    print(f"baseline accuracy\t {baseline_accuracy}")
    print_size_of_model(model, architecture)
    print(f"ptq accuracy\t {ptq_accuracy}")
    print_size_of_model(ptq_model, architecture)
    print(f"qat accuracy\t {qat_accuracy}")
    print_size_of_model(qat_model, architecture)
    print(f"analog accuracy\t {analog_accuracy}")
    print(f"finetuned accuracy\t {finetuned_accuracy}")
    print_size_of_model(analog_model, architecture)


    results_dict = {
        'architecture': str(architecture),

        'baseline_accuracy': baseline_accuracy,
        'ptq_accuracy': ptq_accuracy,
        'qat_accuracy': qat_accuracy,
        'analog_accuracy': analog_accuracy,
        'finetuned_accuracy': finetuned_accuracy,

        'baseline_drift_60': f"{drifted_test_accs_baseline[0].mean().item():.2f} ± {drifted_test_accs_baseline[0].std().item():.2f}",
        'baseline_drift_3600': f"{drifted_test_accs_baseline[1].mean().item():.2f} ± {drifted_test_accs_baseline[1].std().item():.2f}",
        'baseline_drift_86400': f"{drifted_test_accs_baseline[2].mean().item():.2f} ± {drifted_test_accs_baseline[2].std().item():.2f}",
        'baseline_drift_2592000': f"{drifted_test_accs_baseline[3].mean().item():.2f} ± {drifted_test_accs_baseline[3].std().item():.2f}",

        'finetuned_drift_60': f"{drifted_test_accs_finetuned[0].mean().item():.2f} ± {drifted_test_accs_finetuned[0].std().item():.2f}",
        'finetuned_drift_3600': f"{drifted_test_accs_finetuned[1].mean().item():.2f} ± {drifted_test_accs_finetuned[1].std().item():.2f}",
        'finetuned_drift_86400': f"{drifted_test_accs_finetuned[2].mean().item():.2f} ± {drifted_test_accs_finetuned[2].std().item():.2f}",
        'finetuned_drift_2592000': f"{drifted_test_accs_finetuned[3].mean().item():.2f} ± {drifted_test_accs_finetuned[3].std().item():.2f}"

    }

    save_results_safely(results_dict, architecture, rank)


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")

    dist.destroy_process_group()
    sys.stdout.close()
    sys.stderr.close()

def main():
    world_size = int(os.getenv("SLURM_NTASKS", "1"))
    rank = int(os.getenv("SLURM_PROCID", "0"))

    print(world_size)
    print(rank)

    archs = [       
	 	(3,2,1,2,2,1),
	 	(3,2,1,2,2,3),
	 	(3,2,1,2,2,4),
	 	(3,2,2,2,4,3),
	 	(3,2,3,3,1,2),
	 	(3,2,4,1,1,4),
	 	(3,2,4,1,2,1),
	 	(3,2,4,1,2,3),
	 	(3,2,4,1,3,0),
	 	(3,3,0,1,3,3),
	 	(3,3,0,3,3,2),
	 	(3,3,1,1,2,2),
	 	(3,3,2,2,3,1),
	 	(3,3,3,1,2,3),
	 	(3,3,3,2,4,0),
	 	(3,3,3,2,4,2),
	 	(3,3,3,2,4,4),
	 	(3,3,3,3,0,1),
	 	(3,3,3,3,3,3),
	 	(3,3,4,0,3,2),
	 	(3,3,4,0,3,4),
	 	(3,3,4,0,4,1),
	 	(3,3,4,0,4,3),
	 	(3,3,4,3,4,1),
	 	(3,4,0,1,2,3),
	 	(3,4,0,2,4,4),
	 	(3,4,0,3,1,0),
	 	(3,4,1,1,3,1),
	 	(3,4,1,1,3,3),
	 	(3,4,1,2,3,0),
	 	(3,4,1,3,1,2),
	 	(3,4,1,3,1,3)
    ]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")
    print(f"gpus availible : {num_gpus}")

    architecture = archs[rank % len(archs)]
    train(rank, world_size, architecture)

if __name__ == "__main__":
    main()   