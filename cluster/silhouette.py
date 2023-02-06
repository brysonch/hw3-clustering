'''
import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        nclusters = np.amax(y) + 1
        nobs, nlabels = X.shape
        labels = np.arange(0, np.amax(y) + 1)
        a = np.zeros(nobs)

        for cluster in labels:            

            cluster_type = X[cluster == y]
            inds = np.where(cluster == y)
            dists = cdist(cluster_type, cluster_type, metric="euclidean")
            
            a[inds] = np.sum(dists, axis=1) / (dists.shape[0] - 1)

        b = np.zeros(nobs)

        for cluster in labels:
            cluster_type_og = X[cluster == y]
            inds = np.where(cluster == y)
            #other_labels = np.delete(labels, inds)
            other_labels = np.delete(y, inds)
            min_dists = np.full(X[cluster == y].shape[0], np.inf)
            
            for other_cluster in other_labels:
                #print("other labels: ", other_labels)
                cluster_type = X[other_cluster == y]
                #print("other cluster type: ", cluster_type)

                #dists = cdist(cluster_type, cluster_type, metric="euclidean")
                dists = cdist(cluster_type_og, cluster_type, metric="euclidean")
                #print("dists: ", dists)
                avg_dists_per_cluster = np.mean(dists, axis=1)
                min_dists = np.minimum(avg_dists_per_cluster, min_dists)
                # make sure that the orientation of the matrix is correct

            b[inds] = min_dists

        return (b - a) / (np.maximum(a, b))


            #dists.ravel()[::dists.shape[1]+1] = dists.max()+1
            #test[dists.argmin(1)]

            #for obs in range(obs_in_cluster):
                # do not include pt itself
                #cdist(cluster_type[obs], cluster_type, metric="euclidean")

'''
import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        nclusters = np.amax(y) + 1
        nobs, nlabels = X.shape
        labels = np.arange(0, np.amax(y) + 1)
        a = np.zeros(nobs)

        for cluster in np.unique(labels):            

            cluster_type = X[cluster == y]
            inds = np.where(cluster == y)
            dists = cdist(cluster_type, cluster_type, metric="euclidean")
            
            a[inds] = np.sum(dists, axis=1) / (dists.shape[0] - 1)

        b = np.zeros(nobs)

        #for cluster in labels:
        for cluster in np.unique(labels):
            cluster_type_og = X[cluster == y]
            inds = np.where(cluster == y)
            #other_labels = np.delete(labels, inds)
            other_labels = np.delete(y, inds)
            #print("shape of ")
            min_dists = np.full(X[cluster == y].shape[0], np.inf)
            #print("shape: ", other_labels)
            #print("num in this cluster: ", cluster)
            
            #for other_cluster in range(len(other_labels)):
            for other_cluster in np.unique(other_labels):
                #print("num in others : ", other_cluster)
                #print("other labels: ", other_labels)
                cluster_type = X[other_cluster == y]
                #print("other cluster type: ", cluster_type)

                #dists = cdist(cluster_type, cluster_type, metric="euclidean")
                #print("shape of og: ", cluster_type_og.shape)
                #print("shape of later: ", cluster_type.shape)
                dists = cdist(cluster_type_og, cluster_type, metric="euclidean")
                #print("dists: ", dists)
                avg_dists_per_cluster = np.mean(dists, axis=1)
                min_dists = np.minimum(avg_dists_per_cluster, min_dists)
                # make sure that the orientation of the matrix is correct
            
            b[inds] = min_dists
            
        #print("b: ", b)
        #print("a: ", a)
        return (b - a) / (np.maximum(a, b))


