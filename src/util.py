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

    ret = None

    for index in indices:

        i = index[0]
        val = vals[i]

        if i != n - 1:

            test = (arr[:, i:-(n - i) + 1] == val)

        else:

            test = (arr[:, i: ] == val)

        if ret is not None:

            ret = np.logical_and(test, ret)

        else: 

            ret = test

    return ret


def check_matched_indices(pos, indices, check, used_tags, perm, n):

    matches = []
    counts = []

    x = []

    for i in range(n - 1):

        t = pos[i][perm][indices]
        t = list(t[i][check[i]] for i in range(len(check)))
        x.append(t)

    x = np.array(x).T

    for i in range(len(used_tags)):

        e = np.all(x == used_tags[i], axis = 1)
        count = np.sum(e)

        matches.append(e)
        counts.append(count)

    return matches, counts

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


