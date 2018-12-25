import numpy as np
import collections


def ndargmax(arr):
    """
    :param arr: 
    :return: index of maximum element in arr
    """
    return np.unravel_index(np.argmax(arr), arr.shape)

def flatten(x):
    """
    flattens all elements in x
    :param x:
    :return:
    """
    if not isinstance(x, collections.Iterable):
        return np.array([x])
    lst = []
    [lst.extend(e) if isinstance(e, collections.Iterable) else lst.append(e) for e in x]
    return np.array(lst)