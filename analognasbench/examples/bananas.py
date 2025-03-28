import random
import numpy as np
from collections import defaultdict

from sklearn.neural_network import MLPRegressor
from analognasbench.data import AnalogNASBench, DriftMeasurement

def architecture_to_feature_vector(arch):
    return np.array(list(arch), dtype=float)


def evaluate_architecture(benchmark, arch, metric='analog_drift_86400'):
    result = benchmark.query_metric(arch, metric)
    if isinstance(result, DriftMeasurement):
        return result.value
    elif isinstance(result, (int, float)):
        return float(result)
    else:
        return None


def train_ensemble(X, y, n_models=5, hidden_layer_sizes=(32, 32), max_iter=200):
    """
    Train an ensemble of MLP regressors on data (X, y).
    :param X: np.array of shape (n_samples, n_features)
    :param y: np.array of shape (n_samples,)
    :param n_models: number of networks in the ensemble
    :param hidden_layer_sizes: MLP hidden layers
    :param max_iter: training iterations for MLP
    :return: list of trained MLP models
    """
    ensemble = []
    for _ in range(n_models):
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random.randint(0, 999999)
        )
        mlp.fit(X, y)
        ensemble.append(mlp)
    return ensemble


def predict_with_ensemble(ensemble, x):
    """
    Given an ensemble of MLPs and a single feature vector x,
    return the mean and std of predictions across the ensemble.
    """
    preds = [model.predict([x])[0] for model in ensemble]
    preds = np.array(preds)
    return preds.mean(), preds.std()


def bananas_search_analog_nasbench(
    init_size=10,
    total_queries=50,
    metric='analog_drift_86400',
    n_ensemble_models=5,
    alpha=1.0,
    hidden_layer_sizes=(32, 32),
    random_seed=42
):
    """
    A simplified BANANAS-like search to maximize 'analog_drift_86400'.

    :param init_size: How many architectures we randomly sample initially
    :param total_queries: How many total architectures we will evaluate
    :param metric: Which metric from the benchmark to optimize
    :param n_ensemble_models: Number of neural networks in the ensemble
    :param alpha: The weight of the std in the acquisition (mu + alpha * std)
    :param hidden_layer_sizes: Hidden layer sizes for the MLPRegressor
    :param random_seed: Random seed for reproducibility
    :return: (best_arch, best_perf, history)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    benchmark = AnalogNASBench()
    arch_list = benchmark.list_available_architectures()
    if not arch_list:
        print("No architectures found in the benchmark.")
        return None, None, []
    
    init_archs = random.sample(arch_list, min(init_size, len(arch_list)))

    X_data = []
    y_data = []
    tested_set = set()  

    for arch in init_archs:
        perf = evaluate_architecture(benchmark, arch, metric=metric)
        if perf is not None:
            X_data.append(architecture_to_feature_vector(arch))
            y_data.append(perf)
            tested_set.add(arch)

    best_arch = None
    best_perf = float('-inf')

    if X_data:
        max_idx = np.argmax(y_data)
        best_arch_encoded = X_data[max_idx]
        best_perf = y_data[max_idx]

        for i, arch in enumerate(init_archs):
            if i == max_idx:
                best_arch = arch
                break

    history = [] 

    for arch, perf in zip(init_archs, y_data):
        history.append((arch, perf))

    for iteration in range(total_queries - len(tested_set)):
        X_data_arr = np.array(X_data)
        y_data_arr = np.array(y_data)
        ensemble = train_ensemble(
            X_data_arr,
            y_data_arr,
            n_models=n_ensemble_models,
            hidden_layer_sizes=hidden_layer_sizes
        )

        candidate_scores = []
        for arch in arch_list:
            if arch in tested_set:
                continue
            x = architecture_to_feature_vector(arch)
            mu, sigma = predict_with_ensemble(ensemble, x)
            acquisition_value = mu + alpha * sigma
            candidate_scores.append((arch, acquisition_value))

        if not candidate_scores:
            break

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        next_arch = candidate_scores[0][0]

        next_perf = evaluate_architecture(benchmark, next_arch, metric)
        tested_set.add(next_arch)

        if next_perf is not None:
            X_data.append(architecture_to_feature_vector(next_arch))
            y_data.append(next_perf)
            history.append((next_arch, next_perf))
            if next_perf > best_perf:
                best_perf = next_perf
                best_arch = next_arch

        print(f"Iteration {iteration+1}:")
        print(f"\tPicked arch: {next_arch}")
        print(f"\tPerformance: {next_perf}")
        print(f"\tBest so far: {best_arch} with perf={best_perf}")

    return best_arch, best_perf, history


if __name__ == "__main__":
    best_arch, best_val, search_history = bananas_search_analog_nasbench(
        init_size=10,
        total_queries=50,
        metric='analog_drift_86400',
        n_ensemble_models=5,
        alpha=1.0,
        hidden_layer_sizes=(32, 32),
        random_seed=42
    )

    print("\nBANANAS-like Search Complete!")
    print("Best Architecture Found:", best_arch)
    print("Best Performance Value:", best_val)

    benchmark = AnalogNASBench()
    details = benchmark.get_architecture_details(best_arch)
    print("\nFull Architecture Details:")
    print(details)
