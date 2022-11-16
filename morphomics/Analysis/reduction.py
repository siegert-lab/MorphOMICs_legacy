"""
morphomics : dimensionality reduction tools

Author: Ryan Cubero
"""
import numpy as np
import pickle as pkl
import concurrent.futures

import anndata
import palantir
import pandas as pd
from fa2 import ForceAtlas2
from scipy.linalg import svd
from scipy.spatial.distance import cdist, squareform
import umap

from morphomics.Topology import analysis
from morphomics.utils import save_obj, load_obj
from morphomics.utils import norm_methods, distances

scipy_metric = {
    "l1": "cityblock",
    "l2": "euclidean",
}

defaults = {}
# define default image parameter values
defaults['image_parameters'] = {}
defaults['image_parameters']["xlims"] = None
defaults['image_parameters']["ylims"] = None
defaults['image_parameters']["norm_method"] = "sum"
defaults['image_parameters']["metric"] = "l1"
defaults['image_parameters']["chunks"] = 10
defaults['image_parameters']["cutoff"] = 5

defaults['UMAP_parameters'] = {}
defaults['UMAP_parameters']["N_dims"] = 10
defaults['UMAP_parameters']["n_neighbors"] = 20
defaults['UMAP_parameters']["min_dist"] = 1.0
defaults['UMAP_parameters']["spread"] = 3.0
defaults['UMAP_parameters']["random_state"] = 10

def _get_persistence_image_data_single(ar):
    """
    ar[0]:   persistence barcode
    ar[1,2]: x, y-lims
    ar[3]:   normalization method (see morphomics.utils.norm_methods)
    ar[4]:   barcode size cutoff
    """
    if len(ar[0]) >= ar[4]:
        res = analysis.get_persistence_image_data(
            ar[0], xlims=ar[1], ylims=ar[2], norm_method=ar[3]
        )
    else:
        res = np.nan
    return res, ar[0]


def _get_pairwise_distance_from_persistence(
    imgs1, metric="l1", chunks=10, to_squareform=True
):
    """
    Returns mean and spread (standard deviation) of the point cloud of persistence images
    """
    N = len(imgs1)
    distances = np.zeros((N, N))

    # think about chunking this portion of the code
    splits = np.array_split(imgs1, chunks)
    _index = np.array_split(np.arange(N), chunks)
    splits_index = np.hstack(
        [
            0,
            [
                len(np.hstack([_index[i] for i in np.arange(j + 1)]))
                for j in np.arange(len(splits))
            ],
        ]
    )

    for i in np.arange(len(splits)):
        for j in np.arange(i, len(splits)):
            distances[
                splits_index[i] : splits_index[i + 1],
                splits_index[j] : splits_index[j + 1],
            ] = cdist(splits[i], splits[j], metric=scipy_metric[metric])

    # since there are non-zero lower diagonal elements
    distances[np.tril_indices(N)] = 0.0
    # symmetrize distance matrix
    distances = distances + distances.T

    if to_squareform == True:
        return squareform(distances)
    else:
        return distances


def get_images_array(
    p1, xlims=None, ylims=None, norm_method="sum", barcode_size_cutoff=5
):
    """
    Computes persistence images and returns an Nx10000 array,
    N is the number of barcodes in p1
    """
    # get the birth and death distance limits for the persistence images
    _xlims, _ylims = analysis.get_limits(p1)
    if xlims is None:
        xlims = _xlims
    if ylims is None:
        ylims = _ylims

    imgs1 = []
    p1_lims = []
    for p in p1:
        p1_lims.append([p, xlims, ylims, norm_method, barcode_size_cutoff])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # calculate the persistence images in parallel
        for x, y in executor.map(
            _get_persistence_image_data_single, p1_lims, chunksize=200
        ):
            imgs1.append(x)

    N = len(imgs1)
    images = []
    for i in np.arange(N):
        images.append(imgs1[i].flatten() / norm_methods[norm_method](imgs1[i]))

    return images


def get_distance_array(
    p1,
    xlims=None,
    ylims=None,
    norm_method="sum",
    metric="l1",
    chunks=10,
    to_squareform=True,
    barcode_size_cutoff=5,
):
    """
    Computes and outputs array of pre-computed distances for heirarchical clustering
    """
    imgs1 = get_images_array(p1, xlims=xlims, ylims=ylims, norm_method=norm_method, barcode_size_cutoff=barcode_size_cutoff)
    distances = _get_pairwise_distance_from_persistence(
        imgs1, metric=metric, chunks=chunks, to_squareform=to_squareform
    )

    if to_squareform == True:
        return squareform(distances)
    else:
        return distances


# learns coordinates of a force directed layout
def force_directed_layout(
    affinity_matrix, verbose=True, iterations=500, random_seed=10
):
    """ "
    Function to compute force directed layout from the affinity_matrix

    :param affinity_matrix: Sparse matrix representing affinities between cells
    :param cell_names: pandas Series object with cell names
    :param verbose: Verbosity for force directed layout computation
    :param iterations: Number of iterations used by ForceAtlas
    :return: Pandas data frame representing the force directed layout

    Code taken from: https://github.com/dpeerlab/Harmony/blob/master/src/harmony/plot.py
    Added random_seed as an input to control output
    """

    np.random.seed(random_seed)
    init_coords = np.random.random((affinity_matrix.shape[0], 2))

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=False,
        linLogMode=False,
        adjustSizes=False,
        edgeWeightInfluence=1.0,
        # Performance
        jitterTolerance=1.0,
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        # Log
        verbose=verbose,
    )

    positions = forceatlas2.forceatlas2(
        affinity_matrix, pos=init_coords, iterations=iterations
    )
    positions = np.array(positions)

    # Convert to dataframe
    positions = pd.DataFrame(positions, columns=["x", "y"])
    return positions


def calculate_umap(
    barcodes,
    image_parameters=None,
    UMAP_parameters=None,
    save_folder=None,
    save_prefix=None,
):
    """
    Calculates the UMAP representation of the distance matrix X
    """
    # use default if image_parameters is not given
    complete_keys = ["xlims", "ylims", "norm_method", "metric", "chunks", "cutoff"]
    
    if image_parameters is None:
        image_parameters = {}
        
    for keys in complete_keys:
        try:
            image_parameters[keys]
        except:
            print("image_parameters: %s is not given. Reverting to default" % keys)
            image_parameters[keys] = defaults['image_parameters'][keys]

    # calculate distance matrix
    print("Calculating distance matrix...")
    X = get_distance_array(
        barcodes,
        xlims=image_parameters["xlims"],
        ylims=image_parameters["ylims"],
        norm_method=image_parameters["norm_method"],
        metric=image_parameters["metric"],
        chunks=image_parameters["chunks"],
        barcode_size_cutoff=image_parameters["cutoff"],
        to_squareform=True,
    )

    # use default if UMAP_parameters is not given
    complete_keys = ["N_dims", "n_neighbors", "min_dist", "spread", "random_state"]
    
    if UMAP_parameters is None:
        UMAP_parameters = {}
        
    for keys in complete_keys:
        try:
            UMAP_parameters[keys]
        except:
            print("UMAP_parameters: %s is not given. Reverting to default" % keys)
            UMAP_parameters[keys] = defaults['UMAP_parameters'][keys]

    print("Calculating singular value decomposition...")
    U, _, _ = svd(X)
    X_reduced = U.T[0 : UMAP_parameters["N_dims"]]

    print("Calculating UMAP representation...")
    X_umap = umap.UMAP(
        n_neighbors=UMAP_parameters["n_neighbors"],
        min_dist=UMAP_parameters["min_dist"],
        spread=UMAP_parameters["spread"],
        random_state=UMAP_parameters["random_state"],
    ).fit_transform(X_reduced.T)

    if save_folder is not None:
        if save_prefix is None:
            save_prefix = ""
        print("Saving results...")
        save_obj(X_umap, "%s/UMAP_%s" % (save_folder, save_prefix))

    print("Done!")
    return X_umap


def calculate_palantir(X_mat, parameters, save_folder=None, save_prefix=None):
    adata = anndata.AnnData(X=X_mat)

    print("Calculating Palantir maps...")
    pca_projections, _ = palantir.utils.run_pca(
        adata, n_components=parameters["pca_components"], use_hvg=parameters["use_hvg"]
    )

    dm_res = palantir.utils.run_diffusion_maps(
        pca_projections,
        n_components=parameters["dm_components"],
        knn=parameters["knn_components"],
    )

    ms_data = palantir.utils.determine_multiscale_space(dm_res)

    print("Calculating force directed layout...")
    fdl = force_directed_layout(dm_res["kernel"], random_seed=parameters["random_seed"])

    if save_folder is not None:
        if save_prefix is None:
            save_prefix = ""
        print("Saving results...")
        save_obj(fdl, "%s/Palantir_%s" % (save_folder, save_prefix))

    print("Done!")
    return fdl
