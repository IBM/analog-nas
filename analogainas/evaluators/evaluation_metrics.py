import torch
from torch import nn

def negative_mse_metric(dataloader, analog_model,  max_batches=5):
    # This is a metric callback that will be used to evaluate the performance of the model
    # using the mean squared error. The metric is negated because the search algorithm ranks
    # architectures based on the metric.
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