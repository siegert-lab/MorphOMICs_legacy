"""
Contains all the commonly used functions and data
useful for multiple tmd modules.
"""

import pickle as pkl
import numpy as np

term_dict = {"x": 0, "y": 1, "z": 2}

tree_type = {1: "soma", 2: "axon", 3: "basal", 4: "apical"}

array_operators = {
    # make sure that arr is a numpy array
    "<": lambda arr, a: np.array(arr) < a,
    "<=": lambda arr, a: np.array(arr) <= a,
    "==": lambda arr, a: np.array(arr) == a,
    ">=": lambda arr, a: np.array(arr) >= a,
    ">": lambda arr, a: np.array(arr) > a,
}

distances = {
    "l1": lambda t1, t2: np.sum(np.abs(t1 - t2)),
    "l2": lambda t1, t2: np.sqrt(np.dot(t1 - t2, t1 - t2)),
}

norm_methods = {
    # returns the normalization factor based on the method
    "max": lambda arr: np.amax(arr),
    "sum": lambda arr: np.sum(arr),
    "min": lambda arr: np.amin(arr),
    "ave": lambda arr: np.mean(arr),
    "std": lambda arr: np.std(arr),
    "l1": lambda arr: np.sum(np.abs(arr)),
    "l2": lambda arr: np.sqrt(np.dot(arr, arr)),
}


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pkl.load(f)
