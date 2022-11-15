"""
morphomics : bootstrapping tools

Author: Ryan Cubero
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

from morphomics.Analysis.reduction import get_distance_array

bootstrap_methods = {
    "mean": lambda arr: np.mean(arr),
    "median": lambda arr: np.median(arr),
    "max": lambda arr: np.amax(arr),
    "min": lambda arr: np.amin(arr),
    "mean_axis": lambda arr, ax: np.mean(arr, axis=ax),
    "median_axis": lambda arr, ax: np.median(arr, axis=ax),
    "max_axis": lambda arr, ax: np.amax(arr, axis=ax),
    "max_axis": lambda arr, ax: np.amin(arr, axis=ax),
}


def _collect_bars(phs, morphology_list):
    ph_array = []
    for phs_index in morphology_list:
        ph_array.append(phs[phs_index])
    collapsed = [list(pi) for p in ph_array for pi in p]
    return collapsed


def _average_over_features(features, morphology_list, axis=0, method="mean"):
    _features = np.array(features)
    feature_shape = _features.shape

    if len(feature_shape) == 1:
        collapsed = bootstrap_methods[method](_features[morphology_list])
    elif len(feature_shape) > 1:
        collapsed = bootstrap_methods["%s_axis" % method](
            _features[morphology_list], axis
        )
    else:
        raise ValueError("Check the dimension of features...")

    return collapsed


def get_subsampled_population(
    complete_data, N_pop, N_samples, rand_seed, ratio=False, feature="barcodes"
):
    """
    Generates bootstrapped samples of a given feature.

    input:
       complete_data (dict) dictionary containing features. must contain 'barcodes' as one of the keys
       N_pop (int)          bootstrap size
       N_samples (int)      bootstrapped sample size
       rand_seed (int)      seed of the random number generator (to ensure reproducibility)
       ratio (float)        bootstrap-to-population size ratio, 0 < ratio < 1
       feature (str)        feature to bootstrapping on (e.g., barcodes, persistence images, morphometric quantity, ...)

    output:
        subsampled (list)       list of size N_samples containing the bootstrapped feature
        subsampled_index (list) list of size N_samples containing the indices that constituted each bootstrapped feature
    """

    # initialize the random seed
    np.random.seed(rand_seed)

    # create list of non-empty barcodes
    assert complete_data["barcodes"]  # check that 'barcodes' feature exists
    non_empty = np.array(
        [
            ind
            for ind in np.arange(len(complete_data["barcodes"]))
            if len(complete_data["barcodes"][ind]) > 0
        ]
    )

    N = len(non_empty)
    if ratio:  # N_pop=ratio*N if ratio is given
        N_pop = int(ratio * N)
    if N_pop == N:
        print("Nothing to bootstrap.. Returning the original data")
        return np.nan, np.arange(N)
    elif N_pop > N:
        raise ValueError(
            "The bootstrap size is greater than the original population size"
        )

    subsampled = []
    subsampled_index = []

    for kk in np.arange(N_samples):
        ph_array = []
        # draw a subset of cells from the population and save the cell indices
        morphology_list = non_empty[
            np.sort(np.random.choice(np.arange(N), N_pop, replace=False))
        ]
        subsampled_index.append(morphology_list)
        # perform the averaging over the subset of cells and save
        if feature is "barcodes":
            ave_feature = _collect_bars(complete_data[feature], morphology_list)
        else:
            ave_feature = _average_over_features(
                complete_data[feature], morphology_list
            )
        subsampled.append(ave_feature)

    return subsampled, subsampled_index


def _surprise(p):
    if p == 0:
        return 0
    else:
        return -p * np.log2(p)


def _mixing_entropy(clustering, N_samples):
    original_labels = np.array([1] * N_samples + [2] * N_samples)

    p1_size = len(np.where(clustering == 1)[0])
    p2_size = len(np.where(clustering == 2)[0])
    N_ = p1_size + p2_size

    p1 = np.mean(original_labels[np.where(clustering == 1)[0]] == 1)
    p2 = np.mean(original_labels[np.where(clustering == 2)[0]] == 1)

    cluster_entropy = (p1_size / (p1_size + p2_size)) * (
        _surprise(p1) + _surprise(1.0 - p1)
    ) + (p2_size / (p1_size + p2_size)) * (_surprise(p2) + _surprise(1.0 - p2))

    return cluster_entropy


def calculate_mixing_entropy(ph1, ph2, parameters, rand_seed=10):
    N_pop = parameters["N_pop"]
    N_samples = parameters["N_samples"]
    if not parameters["linkage"]:
        parameters["linkage"] = "single"

    phs_cluster_1, _ = get_subsampled_population(ph1, N_pop, N_samples, rand_seed)
    phs_cluster_2, _ = get_subsampled_population(ph2, N_pop, N_samples, rand_seed)

    phs_batched = list(phs_cluster_1) + list(phs_cluster_2)

    X = get_distance_array(phs_batched, xlims=[0, 200], ylims=[0, 200])
    linked = linkage(X, parameters["linkage"], optimal_ordering=False)
    clustering = fcluster(linked, t=2, criterion="maxclust")

    cluster_entropy = _mixing_entropy(clustering)

    return cluster_entropy
