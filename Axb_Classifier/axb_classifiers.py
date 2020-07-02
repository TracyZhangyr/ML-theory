#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from density_radius_clustering import class_split
from hierarchical_clustering import sc_aggl_cluster

np.seterr('ignore')

class axb_classifier:
    """Classifier using linear solver for Ax = b to compare feature vectors

    Parameters
    ----------
    split : {'none', 'density', 'aggl1', 'aggl2'}, default='none'
        Algorithm used to split classes when building
        A matrix.

        Possible values:
        - 'none' : does not split classes when building A matrix
        - 'density' : splits classes by arbitrarily selecting a
          a single data point as the center of a new subclass and
          increasing its radius until the density of points within
          the ball-shaped subclass decreases, repeating this process
          until all points are assigned into a subclass. This process
          is repeated for every class individually. Uses paramters ri,
          dr, and rm.
        - 'aggl1' : uses sklearn's agglomerative clustering algorithm
          to split each class into subclasses independently from one
          another. Uses parameter md.
        - 'aggl2' : implements agglomerative clustering that stops
          merging subclasses when closest subclass comes from a
          different class. To be implemented in the near future.

    ri : float, default=0.1
        Initial radius for 'density' splitting algorithm

    dr : float, default=0.05
        Change in radius between each iteration of the 'density'
        splitting algorithm.

    rm : float, default=1.0
        Maximum radius for any subclass for 'density' splitting
        algorithm.

    md: float, default=1.0
        Maximum distance between two subclasses that can be merged
        for 'aggl1' splitting algorithm.

    Other Attributes
    ----------------

    Am : np.ndarray
        A matrix (from Ax = b)

    """

    def __init__(self, split='none', ri=.1, dr=.05, rm=1.0, md=1.0):
        assert split in {'none', 'density', 'aggl1', 'aggl2'}
        self.split = split
        self.Am = None
        if split == 'density':
            self.subToMain = {}
            self.ri = ri
            self.dr = dr
            self.rm = rm
        elif split == 'aggl1':
            self.subToMain = {}
            self.md = md
        elif split == 'aggl2':
            raise NotImplementedError

    def fit(self, x, y, method='hsm'):
        """Method to build the A matrix.

        Parameters
        ----------
        x : np.ndarray
            Input data numpy array in the form nxm (samplesxfeatures).

        y : np.ndarray
            Numpy array indicating the classes for each sample in x.

        method : {'hsm', 'kde', 'fkde'} 
            String representing how classes (or subclasses) are
            placed into the A matrix.
            Possible values:

            - 'hsm' : places the half-sample mode of each subclass or
              class into the A matrix.
            - 'kde' : places the mode of the kernel density estimation
              representing each subclass or class into the A matrix
            - 'fkde' : similar to KDE except uses fast kernel density
              estimation

        """

        assert method in {'hsm', 'kde', 'fkde'}

        if self.split != 'none':
            x, y = self.splitClass(x, y)
        self.Am = np.zeros((x.shape[1],len(np.unique(y))))
        for feat in range(x.shape[1]):
            for lab in np.unique(y):
                x_fit = x[np.where(y == lab)[0],feat].reshape(-1, 1)
                if method == 'kde':
                    params = {'bandwidth': np.linspace(0.01, 3, 30)}
                    grid = GridSearchCV(KernelDensity(), params, cv=5)
                    grid.fit(x_fit)
                    kde = grid.best_estimator_
                    #h = np.std(x_fit)*(4/3/len(x_fit))**(1/5)
                    #print(h)
                    #kde = KernelDensity(bandwidth=max(h,0.5)).fit(x_fit)
                    self.Am[feat,lab] = x_fit[np.argmax(np.exp(kde.score_samples(x_fit)))]
                elif method == 'hsm':
                    self.Am[feat,lab] = self.hsm(np.sort(x_fit.flatten()))

    def splitClass(self, x, y):
        """Function to separate labelled data into subclasses

        Parameters
        ----------
        x : np.ndarray
            Input data numpy array in the form nxm (samples x features).

        y : np.ndarray
            1-dimensional Numpy array indicating the classes for each sample in x.

        """
        
        label = 0
        splitX = []
        splitY = []
        classes = []

        for lab in np.unique(y):
            subx = x[np.where(y == lab)]
            if self.split == 'density':
                sc = class_split(subx, self.ri, self.dr, self.rm)
            elif self.split == 'aggl1':
                sc = sc_aggl_cluster(subx, self.md)
            elif self.split == 'aggl2':
                raise NotImplementedError
            classes.append(sc)
            start = label
            for subclass in sc:
                splitY += [label] * len(subclass)
                splitX += subclass
                label += 1
            for i in range(start, label):
                self.subToMain[i] = lab

        return np.array(splitX), np.array(splitY)

    @staticmethod
    def hsm(x):
        """Iteratively approximate the mode of a list of floats or integers by dividing
        dataset recursively in half to compare density

        Parameter
        ---------
        x : np.ndarray or [float]
            1-dimensional sequence of real numbers

        """

        n = len(x)

        while len(x) > 3:
            min_width = x[-1] - x[0]
            n = (len(x) + 1) // 2
            min_start = 0
            for i in range(len(x) - n):
                width = x[i+n] - x[i]
                if width < min_width:
                    min_width = width
                    min_start = i
            x = x[min_start:min_start+n]

        if n == 1:
            return x[0]
        elif n == 2:
            return (x[0] + x[1]) / 2
        elif n == 3:
            if x[1] - x[0] < x[2] - x[1]:
                return (x[0] + x[1]) / 2
            elif x[1] - x[0] > x[2] - x[1]:
                return (x[1] + x[2]) / 2
            else:
                return x[1]

    def new_sample_prediction(self, b_mat, pred=None):
        """Solves the Ax=b linear system and returns the class predicted.
        
        Parameters
        ----------

        b_mat : np.ndarray
            b matrix represents the sample(s) from which we want predictions to. Shape should be
            features x samples

        pred : If None the entire vector is returned. Otherwise may be one of the following strings:
            - "HV" so the prediction returned is the Highest Value, 
            - "HM" for the Highest Magnitude, or "LM_1" for the Lowest Magnitude closest to 1.
        """

        assert self.Am.shape[0] == b_mat.shape[0], "The number of features on b (first dimension) doesn't match the first dimension from A."

        output = list(nnls(self.Am, b_mat, rcond=None))
        output[0] = np.atleast_2d(output[0]) if np.ndim(output[0]) < 2 else np.transpose(output[0])
        if not pred:
            return output
        elif pred == "HV":
            return np.argmax(output[0], axis=1)
        elif pred == "HM":
            return np.argmax(np.absolute(output[0]), axis=1)
        elif pred == "LM_1":
            best = list(np.argmin(np.absolute(output[0] - 1), axis=1))
            for i, arg in enumerate(best):
                if abs(output[0][i][arg] - 1) > 1: #label null class
                    best[i] = None
            return best

    def new_sample_prediction_confidence(self, b_mat):
        """Returns confidence that samples in b matrix belong to each class in A matrix

        Parameters
        ----------

        b_mat : np.ndarray
            b matrix represents the sample(s) from which we want predictions to. Shape should be
            features x samples

        """

        if self.split:
            x_matrix = [nnls(self.Am, b_mat.T[i])[0] for i in range(b_mat.shape[1])]
            combined = [[0] * len(self.subToMain) for _ in range(len(x_matrix))]
            for i, row in enumerate(x_matrix):
                for col in range(len(row)):
                    combined[i][self.subToMain[col]] += row[col]

            confidences = list()
            for row in combined:
                temp = [(1/abs(i-1)) for i in row]
                temp = np.exp(temp)/sum(np.exp(temp))
                confidences.append(temp)

        else:
            result = [nnls(self.Am, b_mat.T[i])[0] for i in range(b_mat.shape[1])]
            confidences = list()

            for row in result:
                temp = [(1/np.absolute(i-1)) for i in row]
                temp = np.exp(temp)/sum(np.exp(temp))
                confidences.append(temp)

        return confidences
