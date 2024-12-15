def loss_metric(dataloader, analog_model, criterion):
    losses = []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(self.test_dataloader):
            outputs = analog_model(inputs)
            loss = self.criterion(outputs, targets)

            losses.append(loss.item())