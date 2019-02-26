import numpy as np

def last(arr, low, high, x, n) :

    if high >= low:

        mid = low + (high - low) // 2

        if (mid == n - 1 or x < arr[mid + 1]) and arr[mid] == x:
            
            return mid

        elif x < arr[mid]:

            return last(arr, low, (mid - 1), x, n)

        else :

            return last(arr, (mid + 1), high, x, n)
            
    return -1   

def parse_pos_tags(pos_tags, languages):

    assert(len(pos_tags) <= len(languages))

    x = list(pos_tags[i] for i in range(len(pos_tags))) 

    return np.array(list(languages[i].parse_node(pos_tags[i]) for i in range(len(pos_tags))))

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

