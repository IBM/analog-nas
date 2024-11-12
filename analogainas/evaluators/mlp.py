"""MLP evaluator."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from analognas.evaluators import Evaluator
from analognas.utils import accuracy_mse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


""" 
Base MLP Architecture.
"""
class MLPModel(nn.Module):
    def __init__(
        self,
        input_dims: int = 22,
        num_layers: int = 3,
        layer_width: list = [10, 10, 10],
        output_dims: int = 2,
        activation="relu",
        dropout = 0.0
    ):
        super(MLPModel, self).__init__()

        assert (
            len(layer_width) == num_layers,
            "You need to specify the width of each layer."
        )

        self.activation = eval("F." + activation)

        all_units = [input_dims] + layer_width
        self.layers = nn.ModuleList(
            [nn.Linear(all_units[i], all_units[i + 1])
             for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(all_units[-1], output_dims)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return self.out(x)


"""
MLP Evalutor Wrapper class. 
"""
class MLPEvaluator(Evaluator):
    def __init__(
        self,
        model_type = "MLP",
        hpo_wrapper=False,
        hparams_from_file=False
    ):
        self.model_type = "MLP"
        self.hpo_wrapper = hpo_wrapper
        self.default_hyperparams = {
            "num_layers": 3,
            "layer_width": 10,
            "batch_size": 32,
            "lr": 0.001,
            "dropout": 0.2,
        }
        self.hyperparams = None
        self.hparams_from_file = hparams_from_file

    def get_model(self, **kwargs):
        evaluator = MLPModel(**kwargs)
        return evaluator

    def fit(self, xtrain, ytrain,
            train_info_file="mlp.txt",
            hyperparameters=None,
            epochs=500,
            loss="mae",
            verbose=0):

        if hyperparameters == None:
            self.hyperparams = self.default_hyperparams.copy()
        else:
            self.hyperparams = hyperparameters

        num_layers = self.hyperparams["num_layers"]
        layer_width = self.hyperparams["layer_width"]
        batch_size = self.hyperparams["batch_size"]
        lr = self.hyperparams["lr"]
        dropout = self.hyperparams["dropout"]

        self.mean = np.mean(ytrain)
        self.std = np.std(ytrain)

        scaler = StandardScaler()
        _xtrain = scaler.fit_transform(xtrain)

        _xtrain = xtrain
        _ytrain = np.array(ytrain)

        X_tensor = torch.FloatTensor(_xtrain).to(device)
        y_tensor = torch.FloatTensor(_ytrain).to(device)
        train_data = TensorDataset(X_tensor, y_tensor)

        data_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=False,
        )

        self.model = self.get_model(
            input_dims=_xtrain.shape[1],
            num_layers=num_layers,
            layer_width=num_layers * [layer_width],
        )

        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))

        if loss == "mse":
            criterion = nn.MSELoss().to(device)
        elif loss == "mae":
            criterion = nn.L1Loss().to(device)

        self.model.train()
        with open(train_info_file, "a") as f:
            for e in range(epochs):
                for b, batch in enumerate(data_loader):
                    optimizer.zero_grad()
                    input = batch[0].to(device)
                    target = batch[1].to(device)
                    prediction = self.model(input).view(-1)

                    loss_fn = criterion(prediction, target)
                    params = torch.cat(
                        [
                            x[1].view(-1)
                            for x in self.model.named_parameters()
                            if x[0] == "out.weight"
                        ]
                    )
                    loss_fn += dropout * torch.norm(params, 1)
                    loss_fn.backward()
                    optimizer.step()

                    mse = accuracy_mse(prediction, target)
                    f.write("Loss: {}, MSE: {}\n".format(loss_fn.item(), mse.item()))

                if verbose and e % 100 == 0:
                    f.write("Epoch {}, {}, {}".format(e, loss_fn.item(), mse.item()))

        train_pred = np.squeeze(self.query(xtrain))
        train_error = np.mean(abs(train_pred - ytrain))

        return train_error

    def query(self, xtest, eval_batch_size=None):
        X_tensor = torch.FloatTensor(xtest).to(device)
        test_data = TensorDataset(X_tensor)

        eval_batch_size = len(xtest) if eval_batch_size is None else eval_batch_size
        test_data_loader = DataLoader(
            test_data, batch_size=eval_batch_size, pin_memory=False
        )

        self.model.eval()
        pred = []
        with torch.no_grad():
            for _, batch in enumerate(test_data_loader):
                prediction = self.model(batch[0].to(device)).view(-1)
                pred.append(prediction.cpu().numpy())

        pred = np.concatenate(pred)
        return np.squeeze(pred)

    def set_random_hyperparams(self):
        if self.hyperparams is None:
            params = self.default_hyperparams.copy()

        else:
            params = {
                "num_layers": int(np.random.choice(range(5, 25))),
                "layer_width": int(np.random.choice(range(5, 25))),
                "batch_size": 32,
                "lr": np.random.choice([0.1, 0.01, 0.005, 0.001, 0.0001]),
                "dropout": 0.2,
            }

        self.hyperparams = params
        return params

