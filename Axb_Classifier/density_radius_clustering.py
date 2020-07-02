#!/usr/bin/env python3
# coding: utf-8

import numpy as np

def dist(a, b):
    """Calculates Euclidean distance between 2 points

    Parameters
    ----------

    a : [float] or np.ndarray (1-dimensional)
        Point 1
        
    b : [float] or np.ndarray (1-dimensional)
        Point 2

    """

    assert len(a) == len(b), 'Two points must be in same-dimensional space'
    
    return np.sqrt(sum((a[i] - b[i]) * (a[i] - b[i]) for i in range(len(a))))

def count_members(r, distances):
    """Counts number of distances less than r

    Parameters
    ----------
    r : float
        Maximum distance away from center of circle

    distances : [float] or np.ndarray (1-dimensional)
        Sorted sequence of distances away from center of circle

    """

    for i, d in enumerate(distances):
        if d[0] > r:
            return i

    return len(distances)

def class_split(x, r0, dr, rm):
    """Splits one class into subclasses based on Euclidean distance
    Creates copy of x with extra dimension to represent subclasses

    Parameters
    ----------

    x : [[float]]
        2D list with ordered lists that represent coordinates
    r0 : float
        Initial radius for each subclass
    dr : float
        Change in radius per iteration
    rm : float
        Maximum radius for any subclass

    """

    todo = [[j for j in i] for i in x]
    classes = []
    dim = len(x[0])

    while todo:
        center = todo.pop()
        classs = [center]
        distances = []
        for i, point in enumerate(todo):
            distances.append((dist(center, point), i))
        distances = sorted(distances)
        d0 = 0
        r = r0
        d1 = count_members(r, distances)
        while d0 / r ** dim < d1 / (r + dr) ** dim and r + dr <= rm:
            r += dr
            d0, d1 = d1, count_members(r, distances)
        to_rem = []
        for d, i in distances:
            if d < r:
                to_rem.insert(0, i)
                classs.append(todo[i])
        for i in sorted(to_rem, reverse=True):
            del todo[i]
        classes.append(classs)

    return classes

