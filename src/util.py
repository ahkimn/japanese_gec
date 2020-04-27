# -*- coding: utf-8 -*-

# Filename: util.py
# Date Created: 19/12/2019
# Description: Miscellaneous helper functions
# Python Version: 3.7

import argparse
from ast import literal_eval
import errno
import numpy as np
import os

from itertools import islice, chain

from . import config

cfg = config.parse()
seed = cfg['seed']
RS = np.random.RandomState(seed)


def literal_str(_str):

    if isinstance(_str, str):

        if _str == '\\s':

            return ' '

        elif _str == '\\t':

            return '\t'

        elif _str == '\\n':

            return '\n'

        else:

            return _str

    else:

        raise ValueError('String expected')


def str_list(_str):

    if isinstance(_str, list):

        return _str

    elif isinstance(_str, str):

        return literal_eval(_str)

    else:

        raise ValueError('String or list expected')


def str_bool(_str: str):

    if isinstance(_str, bool):

        return _str

    if _str.lower() in ('yes', 'true', 't', 'y'):

        return True

    elif _str.lower() in ('no', 'false', 'f', 'n'):

        return False

    else:

        raise argparse.ArgumentTypeError('Value of boolean type expected')


def get_files(data_dir: str, filetype: str, n_files: int=-1):
    """
    Obtain a list of files within a directory of a certain filetype

    Args:
        data_dir (str): Directory to search for corpus files
        filetype (str): Filetype to filter for
        n_files (int): Max number of files to return

    Returns:
        (list): List of paths to files satisfying requirements
    """

    # List of compliant files
    file_list = list()

    data_files = os.listdir(data_dir)

    if n_files == -1:

        n_files = len(data_files)

    else:

        n_files = min(len(data_files), n_files)

    perm = RS.permutation(len(data_files))[:n_files]

    for i in range(n_files):

        filename = data_files[perm[i]]

        if filename.endswith(filetype):

            file_list.append(os.path.join(data_dir, filename))

    return file_list


def get_files_recursive(data_dir, filetype):

    ret_list = get_files(data_dir, filetype)

    subdirectories = list(p for p in os.listdir(data_dir) if
                          os.path.isdir(os.path.join(data_dir, p)))

    for sub_dir in subdirectories:
        ret_list += get_files_recursive(os.path.join(data_dir,
                                                     sub_dir), filetype)

    return ret_list


def last(arr, low, high, x, n):
    """
    Recursive search function to find the last occurence of a value within
        a sorted array

    Args:
        arr (arr): Sorted array to search
        low (int): Lowest index to search
        high (int): Highest index to search
        x (int): Value to search for
        n (int): Length of array

    Returns:
        (int): Last index of x in arr or -1 if x is not in arr
    """
    if high >= low:

        # Bisect search region
        mid = low + (high - low) // 2

        # If the value of x occurs at the midpoint
        #   (and the subsequent value is greater than it)
        if (mid == n - 1 or x < arr[mid + 1]) and arr[mid] == x:

            return mid

        # Recursively search prior to midpoint if midpoint greater than x
        elif x < arr[mid]:

            return last(arr, low, (mid - 1), x, n)

        # Recursively search past midpoint if midpoint is less than x
        else:

            return last(arr, (mid + 1), high, x, n)

    # If the value x is not found
    return -1


def search_template(arr, indices, vals, n):
    """
    Perform a rolling search, looking for a template within a given array

    Args:
        arr (np.ndarray): Array to search
        indices (np.ndarray): Indices to match
        vals (np.ndarray): Values to match
        n (int): Length of indices array

    Returns:
        (np.ndarray): Array containing the indices where a sequence of n values
            starting from that index matches the template and values
    """
    ret = None

    for index in indices:

        i = index[0]
        val = vals[i]

        # Restrict search on array to ensure that sequence fits (i.e. search
        #   for first index match should end n-i indices before end
        #   to ensure output has length len(arr) - n + 1)
        if i != n - 1:

            test = (arr[:, i:-(n - i) + 1] == val)

        else:

            test = (arr[:, i:] == val)

        # Perform intersection if ret has already been initialized
        if ret is not None:

            ret = np.logical_and(test, ret)

        # Intialize ret on first iteration
        else:

            ret = test

    return ret


def search_1d(arr, indices, vals, n, _len):
    """
    Perform a rolling search, looking for a template within a given array

    Args:
        arr (np.ndarray): Array to search
        indices (np.ndarray): Indices to match
        vals (np.ndarray): Values to match
        n (int): Length of indices array

    Returns:
        (np.ndarray): Array containing the indices where a sequence of n
            values starting from that index matches the template and values
    """
    len_template = len(vals)
    ret = np.zeros(_len - len_template + 1, dtype=np.bool)

    for i in range(len(ret)):

        match = (arr[i:i + len_template] == vals).reshape(-1)
        ret[i] = np.all(match[indices])

    return ret


def search_2d(arr, vals):
    """
    Perform a rolling search, looking for a template within a given 2D-array

    Args:
        arr (np.ndarray): Array to search
        vals (np.ndarray): Values to match

    Returns:
        (np.ndarray): Array containing the indices where a sequence of n values
            starting from that index matches the template and values
    """
    len_template = len(vals)

    n_match = arr.shape[0]
    n_indices = arr.shape[1]

    n_roll = n_indices - len_template + 1
    ret = np.zeros((n_match, n_roll), dtype=np.bool)

    for i in range(n_roll):

        match = np.all(arr[:, i:i + len_template] == vals, axis=1)
        ret[:, i] = match

    return ret


def search_2d_masked(arr, vals, mask):
    """
    Perform a rolling search, looking for a template within a given 2D-array that is
        masked a boolean array

    Args:
        arr (np.ndarray): Array to search
        vals (np.ndarray): Values to match

    Returns:
        (np.ndarray): Array containing the indices where a sequence of n values
            starting from that index matches the template and values
    """
    len_template = len(vals)

    n_match = arr.shape[0]
    n_indices = arr.shape[1]

    n_roll = n_indices - len_template + 1
    ret = np.zeros((n_match, n_roll), dtype=np.bool)

    non_pad = (arr != 0)

    for i in range(n_roll):

        match = np.ones(n_match, dtype=np.bool)

        for j in range(len_template):

            match = np.logical_and(match, (non_pad[:, i + j]))

            if mask[j]:
                match = np.logical_and(match, (arr[:, i + j] == vals[j]))

        ret[:, i] = match

    return ret


def check_matched_indices(tags, check, possible_tags):
    """
    Determine the type of a match of part-of-speech indices where the types
        are defined by the possible_tags array

    Args:
        pos (np.ndarray): Matrix containing the part-of-speech values to
            search through
        check (np.ndarray): Array determining which indices per row (sentence)
            of pos to search through
        possible_tags (arr): List of tuples containing the possible classes

    Returns:
        (tuple): A tuple containing the following arrays
            matches (arr): An array of np.ndarrays masking where the matched
                phrases align with the each tag of possible_tags
            counts (arr); An array containing the number of matched phrases
                that align with each of possible_tags
    """
    matches = []
    counts = []

    # Array of part-of-speech tags for each matched token
    x = []
    n = possible_tags.shape[1]

    # Iterate over each part-of-speech tags
    for i in range(n):

        templates = tags[:, :, i]
        templates = list(templates[j][check[j]] for j in range(len(check)))
        x.append(templates)

    x = np.array(x).T

    for i in range(len(possible_tags)):

        e = np.all(x == possible_tags[i], axis=1)
        count = np.sum(e)
        matches.append(e)
        counts.append(count)

    return matches, counts


def mkdir_p(path: str, file: bool=False, verbose: bool=False):
    """
    Function to recursively generate directories

    Args:
        path (str): relative pathway to recursively generate
    """
    try:

        if file:

            split_path = os.path.split(path)
            path = split_path[0]

        os.makedirs(path)

        if verbose:
            print('Created directories for path: %s' % path)

    except OSError as exc:

        if exc.errno == errno.EEXIST and os.path.isdir(path):

            if verbose:
                print('Path to %s already exists' % path)
            pass

        else:

            raise


def iter_batch(iterable, size=1):

    while True:
        batch_iter = islice(iterable, size)
        yield chain([next(batch_iter)], batch_iter)


def clear():
    """
    Function to clear the console
    """
    os.system('cls' if os.name == 'nt' else 'clear')
