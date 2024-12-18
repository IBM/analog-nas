import torch
from torch import nn

def negative_mse_metric(dataloader, analog_model,  max_batches=5):
    losses = []
    criterion = nn.MSELoss()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            outputs = analog_model(inputs)
            loss = criterion(outputs, targets)
            loss = -1 * loss.item()

            losses.append(loss)

            if i > max_batches:
                break

    return losses