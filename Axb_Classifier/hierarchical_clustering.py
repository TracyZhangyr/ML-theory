#!/usr/bin/env python3
# coding: utf-8

from sklearn.cluster import AgglomerativeClustering
import numpy as np

def sc_aggl_cluster(data, md):
    """Splits single class into one or more subclasses using
    Sci-kit learn's Agglomerative Clustering class, which separates
    each data point into its own class and merges those closest together.

    Parameters
    ----------
    
    data : [[float]]
        Data from a single class in the form of ordered lists to
        be organized into subclasses

    md : float
        Maximum distance between two subclasses that are being merged

    """

    cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=md,compute_full_tree=True)
    labels = cluster.fit_predict(data)
    sub = []

    for l, d in zip(labels, data):
        while len(sub) <= l:
            sub.append([])
        sub[l].append(d)

    return sub
