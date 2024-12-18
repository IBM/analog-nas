from analogainas.utils import kendal_correlation
import torch

from analogainas.analog_helpers.analog_helpers import create_rpu_config
from aihwkit.nn.conversion import convert_to_analog_mapped
import threading
import os

from torch.utils.tensorboard import SummaryWriter

# Inference Times:
ONE_DAY = 24 * 60 * 60
ONE_MONTH = 30 * ONE_DAY


class RealtimeRPUEvaluator():
    def __init__(self, model, metric_callback, test_dataloader=None, criterion=None, artifact_dir='./results'):
        self.model = model
        self.test_dataloader = test_dataloader
        self.metric_callback = metric_callback
        self.criterion = criterion
        self._arch_string_to_dict = {}
        self._arch_string_to_day_1_metrics = {}
        self._arch_string_to_month_1_metrics = {}

        self._artifact_dir = artifact_dir

        self.analog_inference_device = torch.device("cpu")

        self._tb_log_dir = os.path.join(self._artifact_dir, 'tensorboard')
        os.makedirs(self._tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self._tb_log_dir)
        self.thread_lock = threading.Lock()

        self._global_iteration = 0

    def _get_estimates(self, rpu_architecture, max_batches= 3):
        # Need to swap with metric agnostic version
        model = self.model

        g_max = rpu_architecture['g_max']
        tile_size = rpu_architecture['tile_size']
        dac_res = rpu_architecture['dac_resolution']
        ada_res = rpu_architecture['adc_resolution']

        if str(rpu_architecture) in self._arch_string_to_day_1_metrics:
            return self._arch_string_to_day_1_metrics[str(rpu_architecture)], self._arch_string_to_month_1_metrics[str(rpu_architecture)]

        analog_model = model.to(self.analog_inference_device)
        rpu_config = create_rpu_config(g_max=g_max, tile_size=tile_size, dac_res=dac_res, adc_res=ada_res)

        analog_model = convert_to_analog_mapped(analog_model, rpu_config=rpu_config)

        analog_model.drift_analog_weights(ONE_DAY)

        analog_model.eval()
        day_1_metrics = []

        with torch.no_grad():
            metrics = self.metric_callback(self.test_dataloader, analog_model, max_batches)
            day_1_metrics.extend(metrics)

        analog_model.drift_analog_weights(ONE_MONTH)

        analog_model.eval()
        month_1_metrics = []

        with torch.no_grad():
            metrics = self.metric_callback(self.test_dataloader, analog_model, max_batches)
            month_1_metrics.extend(metrics)

        self._arch_string_to_day_1_metrics[str(rpu_architecture)] = day_1_metrics
        self._arch_string_to_month_1_metrics[str(rpu_architecture)] = month_1_metrics

        return day_1_metrics, month_1_metrics


    def query(self, architecture):
        # Based on the other implementations in this repo,
        # This needs to take in a list of size 1 and returns a tuple of f32 np arrays
        # First output is day 1 measure of performance, second is AVM and isn't used in this repo (query_pop use the second output)
        architecture = architecture[0]
        self._arch_string_to_dict[str(architecture)] = architecture
        #model, training_metrics, validation_metrics = self._get_trained_model(str(architecture))

        day_1_metrics, month_1_metrics = self._get_estimates(str(architecture))
        avg_day_1_metric = sum(day_1_metrics) / len(day_1_metrics)
        avg_month_1_metric = sum(month_1_metrics) / len(month_1_metrics)

        # Needs to become a metric of performance, so we need to negate the metric
        avg_day_1_metric = avg_day_1_metric
        avg_month_1_metric = avg_month_1_metric

        return (avg_day_1_metric, avg_month_1_metric)


    def query_pop(self, architecture_list, should_bypass_eval=False, bypass_threshold=0.0):
        architectures = [a[0] for a in architecture_list]
        for arch in architectures:
            self._arch_string_to_dict[str(arch)] = arch

        day_1_metrics = []
        month_1_metrics = []


        print("Getting estimates")

        for arch in architectures:
            print(str(arch))
            day_1_metric, month_1_metric = self._get_estimates(str(arch))
            avg_day_1_metric = sum(day_1_metric) / len(day_1_metric)
            avg_month_1_metric = sum(month_1_metric) / len(month_1_metric)

            day_1_metrics.append(avg_day_1_metric)
            month_1_metrics.append(avg_month_1_metric)

            print(f"For {arch} day 1 metric: {day_1_metric}, month 1 metric: {month_1_metric}")

        return day_1_metrics, month_1_metrics



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

