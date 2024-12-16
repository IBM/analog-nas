"""Abstract class for base evaluator."""
from analogainas.utils import kendal_correlation
from analogainas.evaluators.base_evaluator import Evaluator
from functools import lru_cache
import torch
from cachetools import LRUCache


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
from torch.multiprocessing import Pool
import threading
import os

from torch.utils.tensorboard import SummaryWriter

# Inference Times:
ONE_DAY = 24 * 60 * 60
ONE_MONTH = 30 * ONE_DAY


"""Class for Evaluating the Model Architecture Directly without an Estimator."""
class RealtimeTrainingEvaluator():
    def __init__(self, model_factory=None, train_dataloader=None, val_dataloader=None, test_dataloader=None, criterion=None, lr = 0.001, epochs=5, patience=4, max_batches=2000, patience_threshold=0.01, gpu_ids=[1,2,3,4,5], artifact_dir='./results'):
        self.model_factory = model_factory
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.lr = lr
        self.epochs = epochs
        self.max_batches = max_batches
        self.patience = patience
        self.patience_threshold = patience_threshold
        self._arch_string_to_dict = {}
        self._model_arch_to_trained_model = LRUCache(maxsize=64)
        self._model_arch_to_training_losses = LRUCache(maxsize=64)
        self._model_arch_to_validation_losses = LRUCache(maxsize=64)
        self._model_arch_to_day_1_losses = LRUCache(maxsize=64)
        self._model_arch_to_month_1_losses = LRUCache(maxsize=64)
        self._artifact_dir = artifact_dir

        self.gpu_ids = gpu_ids
        self.training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analog_inference_device = torch.device("cpu")

        self._tb_log_dir = os.path.join(self._artifact_dir, 'tensorboard')
        os.makedirs(self._tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self._tb_log_dir)
        self.thread_lock = threading.Lock()

        self._global_iteration = 0

    def _train_model_thread(self, architecture_string, device_id):
        gpu_path = self._tb_log_dir + f'/gpu:{device_id}'
        os.makedirs(gpu_path, exist_ok=True)
        gpu_writer = SummaryWriter(log_dir=gpu_path)

        device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")
        architecture = self._arch_string_to_dict[str(architecture_string)]

        model = self.model_factory(architecture)
        model = model.to(device)

        model = model.to(device)
        training_losses = []
        validation_losses = []
        patience_counter = 0

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        print(f"Launching training for {architecture_string} on {device}")

        batch_idx = 0
        for epoch in range(self.epochs):
            # Training
            model.train()
            for i, (inputs, targets) in enumerate(self.train_dataloader):

                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                training_losses.append(loss.item())

                if batch_idx % 100 == 0:
                    print(f'{device} - Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
                    with self.thread_lock:
                        gpu_writer.add_scalar(f'{self._global_iteration}/training_loss', loss.item(), batch_idx)

                if i > self.max_batches:
                    break

                batch_idx += 1

            # Validation
            model.eval()
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(self.val_dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    outputs = model(inputs)
                    loss = self.criterion(outputs, targets)

                    validation_losses.append(loss.item())

            print(f'{device} - Epoch: {epoch}, Training Loss: {training_losses[-1]}, Validation Loss: {validation_losses[-1]}')
            with self.thread_lock:
                gpu_writer.add_scalar(f'{self._global_iteration}/validation_loss', validation_losses[-1], epoch)

            if epoch > 0 and validation_losses[-1] > validation_losses[-2] - self.patience_threshold:
                patience_counter += 1
            if patience_counter >= self.patience:
                break
        print(f'{device} - Done training architecture')
        model.to(self.analog_inference_device)
        self._model_arch_to_trained_model[architecture_string] = model
        self._model_arch_to_training_losses[architecture_string] = training_losses
        self._model_arch_to_validation_losses[architecture_string] = validation_losses

    def _train_architectures_threaded(self, architectures):
        # Create batches of architectures to train based on num of GPUs
        num_gpus = len(self.gpu_ids)
        batches = [architectures[i:i + num_gpus] for i in range(0, len(architectures), num_gpus)]

        # Train each batch of architectures on all gpus
        for batch in batches:
            print(f"Starting new batch with {len(batch)} architectures")
            # with Pool(num_gpus) as p:
            #     p.starmap(self._train_model_thread, [(arch, self.gpu_ids[i]) for i, arch in enumerate(batch)])
            threads = []
            for i, arch in enumerate(batch):
                t = threading.Thread(target=self._train_model_thread, args=(str(arch), self.gpu_ids[i]))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()


    def _get_trained_models(self, architectures):
        trained_models = []
        architectures_to_train = []
        for arch in architectures:
            if str(arch) not in self._model_arch_to_trained_model:
                architectures_to_train.append(arch)
        if len(architectures_to_train) > 0:
            self._train_architectures_threaded(architectures_to_train)
            self._global_iteration += 1
        for arch in architectures:
            trained_models.append(self._model_arch_to_trained_model[str(arch)])
        return trained_models

    def _get_estimates(self, architecture, max_batches= 3):
        # Need to swap with metric agnostic version
        architecture = self._arch_string_to_dict[str(architecture)]
        model = self._model_arch_to_trained_model[str(architecture)]

        if str(architecture) in self._model_arch_to_day_1_losses:
            return self._model_arch_to_day_1_losses[str(architecture)], self._model_arch_to_month_1_losses[str(architecture)]

        analog_model = model.to(self.analog_inference_device)
        analog_model = convert_to_analog_mapped(analog_model, rpu_config=create_rpu_config())

        analog_model.drift_analog_weights(ONE_DAY)

        analog_model.eval()
        day_1_losses = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_dataloader):
                outputs = analog_model(inputs)
                loss = self.criterion(outputs, targets)

                day_1_losses.append(loss.item())

                if i > max_batches:
                    break

            print(f"Day 1 losses: {day_1_losses}")

        analog_model.drift_analog_weights(ONE_MONTH)

        analog_model.eval()
        month_1_losses = []

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(self.test_dataloader):
                outputs = analog_model(inputs)
                loss = self.criterion(outputs, targets)

                month_1_losses.append(loss.item())

                if i > max_batches:
                    break

            print(f"Month 1 losses: {month_1_losses}")

        self._model_arch_to_day_1_losses[str(architecture)] = day_1_losses
        self._model_arch_to_month_1_losses[str(architecture)] = month_1_losses

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
        for arch in architectures:
            self._arch_string_to_dict[str(arch)] = arch

        day_1_losses = []
        month_1_losses = []

        self._get_trained_models(architectures)

        print("Getting estimates")

        for arch in architectures:
            day_1_loss, month_1_loss = self._get_estimates(str(arch))
            avg_day_1_loss = sum(day_1_loss) / len(day_1_loss)
            avg_month_1_loss = sum(month_1_loss) / len(month_1_loss)

            day_1_losses.append(avg_day_1_loss)
            month_1_losses.append(avg_month_1_loss)

            print(f"For {arch} day 1 loss: {day_1_loss}, month 1 loss: {month_1_loss}")

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

