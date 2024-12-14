"""Abstract class for base evaluator."""
from analogainas.utils import kendal_correlation
from analogainas.evaluators.base_evaluator import Evaluator
from functools import lru_cache
import torch


from analogainas.analog_helpers.analog_helpers import create_rpu_config, create_analog_optimizer
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import WeightClipType
from aihwkit.simulator.configs.utils import BoundManagementType
from aihwkit.simulator.presets.utils import PresetIOParameters
from aihwkit.inference.noise.pcm import PCMLikeNoiseModel
from aihwkit.inference.compensation.drift import GlobalDriftCompensation
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn import AnalogSequential
from aihwkit.optim import AnalogSGD

# Inference Times:
ONE_DAY = 24 * 60 * 60
ONE_MONTH = 30 * ONE_DAY


"""Class for Evaluating the Model Architecture Directly without an Estimator."""
class RealtimeTrainingEvaluator():
    def __init__(self, model_factory=None, train_dataloader=None, val_dataloader=None, test_dataloader=None, criterion=None, lr = 0.001, epochs=5, patience=4, patience_threshold=0.01):
        self.model_factory = model_factory
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test__dataloader = test_dataloader
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.patience_threshold = patience_threshold
        self._arch_string_to_dict = {}

    @lru_cache(maxsize=32)
    def _get_trained_model(self, model_arch):
        model_arch = self._arch_string_to_dict[str(model_arch)]
        model = self.model_factory(model_arch)
        training_losses = []
        validation_losses = []
        patience_counter = 0

        optimizer = create_analog_optimizer(model, self.lr)

        for epoch in range(self.epochs):
            # Training
            model.train()
            for i, (inputs, targets) in enumerate(self.train_dataloader):
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                training_losses.append(loss.item())

            # Validation
            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self.val_dataloader):
                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    validation_losses.append(loss.item())

            if epoch > 0 and validation_losses[-1] > validation_losses[-2] - self.patience_threshold:
                patience_counter += 1
            if patience_counter >= self.patience:
                break
        print(f'Done training: {model_arch}')

        return model, training_losses, validation_losses

    @lru_cache(maxsize=32)
    def _get_estimates(self, architecture):
        # Need to swap with metric agnostic version
        architecture = self._arch_string_to_dict[str(architecture)]
        model, training_losses, validation_losses = self._get_trained_model(str(architecture))

        analog_model = model.to(torch.device("cpu"))
        analog_model = convert_to_analog_mapped(analog_model, rpu_config=create_rpu_config())

        analog_model = analog_model.drift_analog_weights(ONE_DAY)

        analog_model.eval()
        day_1_losses = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_dataloader):
                outputs = analog_model(inputs)
                loss = self.criterion(outputs, targets)

                day_1_losses.append(loss.item())

        analog_model = analog_model.drift_analog_weights(ONE_MONTH)

        analog_model.eval()
        month_1_losses = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_dataloader):
                outputs = analog_model(inputs)
                loss = self.criterion(outputs, targets)

                month_1_losses.append(loss.item())

        return day_1_losses, month_1_losses


    def query(self, architecture):
        # Based on the other implementations in this repo,
        # This needs to take in a list of size 1 and returns a tuple of f32 np arrays
        # First output is day 1 measure of performance, second is AVM and isn't used in this repo (query_pop use the second output)
        architecture = architecture[0]
        self._arch_string_to_dict[str(architecture)] = architecture
        #model, training_losses, validation_losses = self._get_trained_model(str(architecture))

        day_1_losses, month_1_losses = self._get_estimates(str(architecture))
        avg_day_1_loss = sum(day_1_losses) / len(day_1_losses)
        avg_month_1_loss = sum(month_1_losses) / len(month_1_losses)

        # Needs to become a metric of performance, so we need to negate the loss
        avg_day_1_loss = -1 * avg_day_1_loss
        avg_month_1_loss = -1 * avg_month_1_loss

        return (avg_day_1_loss, avg_month_1_loss)


    def query_pop(self, architecture_list):
        architectures = [a[0] for a in architecture_list]

        day_1_losses = []
        month_1_losses = []

        for architecture in architectures:
            day_1_loss, month_1_loss = self.query([architecture])
            day_1_losses.append(day_1_loss)
            month_1_losses.append(month_1_loss)

        return day_1_losses, month_1_losses


    def set_hyperparams(self, hyperparams):
        """
        Modifies/sets hyperparameters of the evaluator.

        Args:
            hyperparams: dictionary of hyperparameters.
        """
        self.hyperparams = hyperparams

    def get_hyperparams(self):
        """
        Get the hyperparameters of the evaluator.

        Returns:
            A dictionary of hyperparameters.
            If not manually set, a dictionary of the default hyperparameters.
        """
        if hasattr(self, "hyperparams"):
            return self.hyperparams
        else:
            return None

    def get_correlation(self, x_test, y_test):
        y_pred = self.query(x_test)
        return kendal_correlation(y_test, y_pred)

