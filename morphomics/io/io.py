"""
Python module that contains the functions
about reading and writing files.
"""
from __future__ import print_function

import os, glob
import numpy as _np
from scipy import sparse as sp
from scipy.sparse import csgraph as cs


# The following codes were adapted from TMD:
# https://github.com/BlueBrain/TMD
from morphomics.io.swc import SWC_DCT
from morphomics.io.swc import read_swc
from morphomics.io.swc import swc_to_data
from morphomics.Neuron import Neuron
from morphomics.Tree import Tree
from morphomics.Soma import Soma
from morphomics.Population import Population
from morphomics.utils import tree_type as td
from morphomics.utils import save_obj, load_obj
from morphomics.Topology import analysis
from morphomics.Topology import methods


# Definition of tree types
TYPE_DCT = {"soma": 1, "basal": 3, "apical": 4, "axon": 2, "glia": 7}


class LoadNeuronError(Exception):
    """
    Captures the exception of failing to load a single neuron
    """


def make_tree(data):
    """
    Make tree structure from loaded data.
    Returns a tree of morphomics.Tree type.
    """
    tr_data = _np.transpose(data)

    parents = [
        _np.where(tr_data[0] == i)[0][0]
        if len(_np.where(tr_data[0] == i)[0]) > 0
        else -1
        for i in tr_data[6]
    ]

    return Tree.Tree(
        x=tr_data[SWC_DCT["x"]],
        y=tr_data[SWC_DCT["y"]],
        z=tr_data[SWC_DCT["z"]],
        d=tr_data[SWC_DCT["radius"]],
        t=tr_data[SWC_DCT["type"]],
        p=parents,
    )


def load_neuron(
    input_file,
    line_delimiter="\n",
    soma_type=None,
    tree_types=None,
    remove_duplicates=True,
):
    """
    Io method to load an swc into a Neuron object.
    TODO: Check if tree is connected to soma, otherwise do
    not include it in neuron structure and warn the user
    that there are disconnected components
    """

    tree_types_final = td.copy()
    if tree_types is not None:
        tree_types_final.update(tree_types)

    # Definition of swc types from type_dict function
    if soma_type is None:
        soma_index = TYPE_DCT["soma"]
    else:
        soma_index = soma_type

    # Make neuron with correct filename and load data
    if os.path.splitext(input_file)[-1] == ".swc":
        data = swc_to_data(
            read_swc(input_file=input_file, line_delimiter=line_delimiter)
        )
        neuron = Neuron.Neuron(name=input_file.replace(".swc", ""))

    try:
        soma_ids = _np.where(_np.transpose(data)[1] == soma_index)[0]
    except IndexError:
        raise LoadNeuronError("Soma points not in the expected format")
    print(os.path.splitext(input_file)[-2:], len(soma_ids))

    # Extract soma information from swc
    soma = Soma.Soma(
        x=_np.transpose(data)[SWC_DCT["x"]][soma_ids],
        y=_np.transpose(data)[SWC_DCT["y"]][soma_ids],
        z=_np.transpose(data)[SWC_DCT["z"]][soma_ids],
        d=_np.transpose(data)[SWC_DCT["radius"]][soma_ids],
    )

    # Save soma in Neuron
    neuron.set_soma(soma)
    p = _np.array(_np.transpose(data)[6], dtype=int) - _np.transpose(data)[0][0]
    # return p, soma_ids
    try:
        dA = sp.csr_matrix(
            (
                _np.ones(len(p) - len(soma_ids)),
                (range(len(soma_ids), len(p)), p[len(soma_ids) :]),
            ),
            shape=(len(p), len(p)),
        )
    except Exception:
        raise LoadNeuronError(
            "Cannot create connectivity, nodes not connected correctly."
        )

    # assuming soma points are in the beginning of the file.
    comp = cs.connected_components(dA[len(soma_ids) :, len(soma_ids) :])

    # Extract trees
    for i in range(comp[0]):
        tree_ids = _np.where(comp[1] == i)[0] + len(soma_ids)
        tree = make_tree(data[tree_ids])
        neuron.append_tree(tree, td=tree_types_final)

    return neuron


def load_ph_file(filename, delimiter=" "):
    """
    Load PH file in a np.array
    """
    f = open(filename, "r")
    ph = _np.array([_np.array(line.split(delimiter), dtype=_np.float) for line in f])
    f.close()
    return ph


def load_population(neurons, tree_types=None, name=None):
    """
    Loads all data of recognised format (swc)
    into a Population object.
    Takes as input a directory or a list of files to load.
    """
    if isinstance(neurons, (list, tuple)):
        files = neurons
        name = name if name is not None else "Population"
    elif os.path.isdir(neurons):  # Assumes given input is a directory
        files = [os.path.join(neurons, l) for l in os.listdir(neurons)]
        name = name if name is not None else os.path.basename(neurons)

    pop = Population.Population(name=name)

    fnames = []
    for filename in files:
        try:
            assert filename.endswith((".swc"))
            pop.append_neuron(load_neuron(filename, tree_types=tree_types))
            fnames.append(filename)
        except AssertionError:
            raise Warning("{} is not a valid swc file".format(filename))
        except LoadNeuronError:
            print("File failed to load: {}".format(filename))

    return pop, fnames


def exclude_single_branch_ph(neuron, feature="path_distances"):
    """
    Calculates persistence diagram and only considers
    only those with more than one bar
    """
    phs = []
    for tree in neuron.neurites:
        p = methods.get_persistence_diagram(tree, feature="path_distances")
        if len(p) > 1:
            phs.append(p)
    return phs


def get_clean_input(input_directory, extension=".swc", save_filename=None):
    """
    Loads data contained in input directory that ends in ".swc".

    Input


    Returns all the morphologies (pops), barcodes (phs) and filenames (files)
    in a dictionary , and will be saved if save_filename is given.
    """
    filename = glob.glob("%s/*%s" % (input_directory, extension))
    pops, files = load_population(filename)

    phs = [
        analysis.collapse(exclude_single_branch_ph(n, feature="path_distances"))
        for n in pops.neurons
    ]

    data = {}
    data["morphologies"] = pops
    data["barcodes"] = phs
    data["filenames"] = files

    if save_filename is not None:
        save_obj(data, save_filename)

    return data


def combine_data(list_of_data, keys_to_combine, save_filename=None):
    """
    Inputs a collection of datasets extracted from get_clean_input

    list_of_data: list of results from get_clean_input
    keys_to_combine: list of the keys that will be combined
    save_filename: full filename where to save the combined data

    Returns combined data as a dictionary, and will be saved
    if save_filename is given.
    """
    assert type(list_of_data) is list, "Input data is not a list"
    assert len(list_of_data) > 1, "There is only one data. No need to combine"

    _combs = {}
    for keys in keys_to_combine:
        _combs[keys] = []

    # combine the datasets here
    for data in list_of_data:
        for keys in keys_to_combine:
            _combs[keys] = _combs[keys] + data[keys]

    combined_data = {}
    for keys in keys_to_combine:
        combined_data[keys] = _combs[keys]

    if save_filename is not None:
        save_obj(combined_data, save_filename)

    return combined_data
