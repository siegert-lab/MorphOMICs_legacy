"""
morphomics Topology analysis algorithms implementation
Adapted from https://github.com/BlueBrain/TMD
"""
# pylint: disable=invalid-slice-index
import copy
import math
from itertools import chain
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cdist
from scipy import stats
from morphomics.utils import array_operators as ops
from morphomics.utils import norm_methods


def get_lengths(ph):
    """
    Returns the lengths of the bars from the diagram
    """
    return np.array([np.abs(i[0] - i[1]) for i in ph])


def collapse(ph_list):
    """
    Collapses a list of ph diagrams
    into a single instance for plotting.
    """
    return [list(pi) for p in ph_list for pi in p]


def sort_ph(ph):
    """
    Sorts barcode according to decreasing length of bars.
    """
    return np.array(ph)[np.argsort([p[0] - p[1] for p in ph])].tolist()


def filter_ph(ph, cutoff, method="<="):
    """
    Cuts off bars depending on their length
    ph:
    cutoff:
    methods: "<", "<=", "==", ">=", ">"
    """
    barcode_length = []
    if len(ph) >= 1:
        lengths = get_lengths(ph)
        cut_offs = np.where(ops[method](lengths, cutoff))[0]
        if len(cut_offs) >= 1:
            barcode_length.append(barcodes[cut_offs])

        return barcode_length

    else:
        raise "Barcode is empty"


def closest_ph(ph_list, target_extent, method="from_above"):
    """
    Returns the index of the persistent homology in the ph_list that has the maximum extent
    which is closer to the target_extent according to the selected method.

    method:
        from_above: smallest maximum extent that is greater or equal than target_extent
        from_below: biggest maximum extent that is smaller or equal than target_extent
        nearest: closest by absolute value
    """
    n_bars = len(ph_list)
    max_extents = np.asarray([max(get_lengths(ph)) for ph in ph_list])

    sorted_indices = np.argsort(max_extents, kind="mergesort")
    sorted_extents = max_extents[sorted_indices]

    if method == "from_above":

        above = np.searchsorted(sorted_extents, target_extent, side="right")

        # if target extent is close to current one, return this instead
        if above >= 1 and np.isclose(sorted_extents[above - 1], target_extent):
            closest_index = above - 1
        else:
            closest_index = above

        closest_index = np.clip(closest_index, 0, n_bars - 1)

    elif method == "from_below":

        below = np.searchsorted(sorted_extents, target_extent, side="left")

        # if target extent is close to current one, return this instead
        if below < n_bars and np.isclose(sorted_extents[below], target_extent):
            closest_index = below
        else:
            closest_index = below - 1

        closest_index = np.clip(closest_index, 0, n_bars - 1)

    elif method == "nearest":

        below = np.searchsorted(sorted_extents, target_extent, side="left")
        pos = np.clip(below, 0, n_bars - 2)

        closest_index = min(
            (pos, pos + 1), key=lambda i: abs(sorted_extents[i] - target_extent)
        )

    else:
        raise TypeError("Unknown method {} for closest_ph".format(method))

    return sorted_indices[closest_index]


def get_limits(phs_list, coll=True):
    """Returns the x-y coordinates limits (min, max)
    for a list of persistence diagrams
    """
    if coll:
        ph = collapse(phs_list)
    else:
        ph = copy.deepcopy(phs_list)
    xlims = [min(np.transpose(ph)[0]), max(np.transpose(ph)[0])]
    ylims = [min(np.transpose(ph)[1]), max(np.transpose(ph)[1])]
    return xlims, ylims


def get_persistence_image_data(
    ph, xlims=None, ylims=None, norm_factor=None, bw_method=None, norm_method="max"
):
    """
    Create the data for the generation of the persistence image.
    ph: persistence diagram
    norm_factor: persistence image data are normalized according to this.
        If norm_factor is provided the data will be normalized based on this,
        otherwise they will be normalized to 1.
    xlims, ylims: the image limits on x-y axes.
        If xlims, ylims are provided the data will be scaled accordingly.
    bw_method: The method used to calculate the estimator bandwidth for the gaussian_kde.
    norm_method: The method used to normalize the persistence images (chosen between "max" or "sum")
    """
    if xlims is None or xlims is None:
        xlims, ylims = get_limits(ph, coll=False)

    X, Y = np.mgrid[xlims[0] : xlims[1] : 100j, ylims[0] : ylims[1] : 100j]

    values = np.transpose(ph)
    kernel = stats.gaussian_kde(values, bw_method=bw_method)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kernel(positions).T, X.shape)

    if norm_factor is None:
        norm_factor = norm_methods[norm_method](Z)

    return Z / norm_factor


def get_image_diff_data(Z1, Z2, normalized=True, norm_method="max"):
    """
    Takes as input two images as exported from the gaussian kernel
    plotting function, and returns their difference: diff(Z1 - Z2)
    """
    if normalized:
        Z1_norm = norm_methods[norm_method](Z1)
        Z2_norm = norm_methods[norm_method](Z2)

        Z1 = Z1 / Z1_norm
        Z2 = Z2 / Z2_norm
    return Z1 - Z2


def get_image_add_data(Z1, Z2, normalized=True, norm_method="max"):
    """
    Takes as input two images
    as exported from the gaussian kernel
    plotting function, and returns
    their sum: add(Z1 - Z2)
    """
    if normalized:
        Z1_norm = norm_methods[norm_method](Z1)
        Z2_norm = norm_methods[norm_method](Z2)

        Z1 = Z1 / Z1_norm
        Z2 = Z2 / Z2_norm
    return Z1 + Z2


def get_average_persistence_image(
    ph_list, xlims=None, ylims=None, norm_factor=None, weighted=False
):
    """
    Plots the gaussian kernel of a population of cells
    as an average of the ph diagrams that are given.
    """
    im_av = False
    k = 1
    if weighted:
        weights = [len(p) for p in ph_list]
        weights = np.array(weights, dtype=np.float) / np.max(weights)
    else:
        weights = [1 for _ in ph_list]

    for weight, ph in zip(weights, ph_list):
        if not isinstance(im_av, np.ndarray):
            try:
                im = get_persistence_image_data(
                    ph, norm_factor=norm_factor, xlims=xlims, ylims=ylims
                )
                if not np.isnan(np.sum(im)):
                    im_av = weight * im
            except BaseException:  # pylint: disable=broad-except
                pass
        else:
            try:
                im = get_persistence_image_data(
                    ph, norm_factor=norm_factor, xlims=xlims, ylims=ylims
                )
                if not np.isnan(np.sum(im)):
                    im_av = np.add(im_av, weight * im)
                    k = k + 1
            except BaseException:  # pylint: disable=broad-except
                pass
    return im_av / k


def find_apical_point_distance(ph):
    """
    Finds the apical distance (measured in distance from soma)
    based on the variation of the barcode.
    """
    # Computation of number of components within the barcode
    # as the number of bars with at least max length / 2
    lengths = get_lengths(ph)
    num_components = len(np.where(np.array(lengths) >= max(lengths) / 2.0)[0])
    # Separate the barcode into sufficiently many bins
    n_bins, counts = histogram_horizontal(ph, num_bins=3 * len(ph))
    # Compute derivatives
    der1 = counts[1:] - counts[:-1]  # first derivative
    der2 = der1[1:] - der1[:-1]  # second derivative
    # Find all points that take minimum value, defined as the number of components,
    # and have the first derivative zero == no variation
    inters = np.intersect1d(
        np.where(counts == num_components)[0], np.where(der1 == 0)[0]
    )
    # Find all points that are also below a positive second derivative
    # The definition of how positive the second derivative should be is arbitrary,
    # but it is the only value that works nicely for cortical cells
    try:
        best_all = inters[
            np.where(inters <= np.max(np.where(der2 > len(n_bins) / 100)[0]))
        ]
    except ValueError:
        return 0.0

    if len(best_all) == 0 or n_bins[np.max(best_all)] == 0:
        return np.inf
    return n_bins[np.max(best_all)]


def find_apical_point_distance_smoothed(ph, threshold=0.1):
    """
    Finds the apical distance (measured in distance from soma)
    based on the variation of the barcode.
    This algorithm always computes a distance, even if
    there is no obvious apical point.
    Threshold corresponds to percent of minimum derivative variation
    that is used to select the minima.
    """
    bin_centers, data = barcode_bin_centers(ph, num_bins=100)

    # Gaussian kernel to smooth distribution of bars
    kde = stats.gaussian_kde(data)
    minimas = []
    while len(minimas) == 0:
        # Compute first derivative
        der1 = np.gradient(kde(bin_centers))
        # Compute second derivative
        der2 = np.gradient(der1)
        # Compute minima of distribution
        minimas = np.where(abs(der1) < threshold * np.max(abs(der1)))[0]
        minimas = minimas[der2[minimas] > 0]
        threshold *= 2.0  # if threshold was too small, increase and retry
    return bin_centers[minimas[0]]


def _symmetric(p):
    """
    Returns the symmetric point of a PD point on the diagonal
    """
    return [(p[0] + p[1]) / 2.0, (p[0] + p[1]) / 2]
