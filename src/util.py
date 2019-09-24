# Filename: load.py
# Author: Alex Kimn
# E-mail: alex.kimn@outlook.com
# Date Created: 18/06/2018
# Date Last Modified: 27/02/2019
# Python Version: 3.7

'''
Miscellaneous helper functions
'''

import numpy as np
import os


def last(arr, low, high, x, n):
    """
    Recursive search function to find the last occurence of a value within a sorted array
    
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

        # If the value of x occurs at the midpoint (and the subsequent value is greater than it)
        if (mid == n - 1 or x < arr[mid + 1]) and arr[mid] == x:
            
            return mid

        # Recursively search prior to midpoint if midpoint greater than x
        elif x < arr[mid]:

            return last(arr, low, (mid - 1), x, n)

        # Recursively search past midpoint if midpoint is less than x
        else :

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
        (np.ndarray): Array containing the indices where a sequence of n values starting from that index matches the template and values
    """
    ret = None

    for index in indices:

        i = index[0]
        val = vals[i]

        # Restrict search on array to ensure that sequence fits (i.e. search for first index match should end n-i indices before end
        # to ensure output has length len(arr) - n + 1)
        if i != n - 1:

            test = (arr[:, i:-(n - i) + 1] == val)

        else:

            test = (arr[:, i: ] == val)

        # Perform intersection if ret has already been initialized
        if ret is not None:

            ret = np.logical_and(test, ret)

        # Intialize ret on first iteration
        else: 

            ret = test

    return ret


def check_matched_indices(pos, check, used_tags):
    """
    Determine the type of a match of part-of-speech indices where the types are defined by the used_tags array
    
    Args:
        pos (np.ndarray): Matrix containing the part-of-speech values to search through
        check (np.ndarray): Array determining which indices per row (sentence) of pos to search through
        used_tags (arr): List of tuples containing the possible classes
    
    Returns:
        (tuple): A tuple containing the following arrays
            matches (arr): An array of np.ndarrays masking where the matched phrases align with the each tag of used_tags
            counts (arr); An array containing the number of matched phrases that align with each of used_tags
    """
    matches = []
    counts = []

    # Array of part-of-speech tags for each matched token
    x = []
    n = pos.shape[0]

    # Iterate over each part-of-speech tags
    for i in range(n):

        templates = pos[i]
        templates = list(templates[j][check[j]] for j in range(len(check)))
        x.append(templates)

    x = np.array(x).T

    for i in range(len(used_tags)):

        e = np.all(x == used_tags[i], axis = 1)
        count = np.sum(e)

        matches.append(e)
        counts.append(count)

    return matches, counts


def get_files(data_dir, filetype, n_files=-1):
    """
    Obtain a list of files within a directory of a certain filetype
    
    Args:
        data_dir (str): Directory to search for corpus files
        filetype (str): File suffix
        n_files (int): Max number of files to obtain
    
    Returns:
        (arr): List of paths to files satisfying requirements
    """
    # List of compliant files
    file_list = list()

    data_files = os.listdir(data_dir)
    print(data_files)
    print(filetype)

    if n_files == -1:

        n_files = len(data_files)

    else:

        n_files = min(len(data_files), n_files)

    perm = np.random.permutation(len(data_files))[:n_files]

    for i in range(n_files):

        filename = data_files[perm[i]]

        if filename.endswith(filetype):

            file_list.append(os.path.join(data_dir, filename))

    return file_list


def mkdir_p(path):
    """    
    Function to recursively generate directories

    Args:
        path (str): relative pathway to recursively generate
    """
    try:

        os.makedirs(path)

    except OSError as exc:

        if exc.errno == errno.EEXIST and os.path.isdir(path):

            pass

        else:

            raise


def clear_():
    """
    Function to clear the console
    """
    os.system('cls' if os.name == 'nt' else 'clear')
