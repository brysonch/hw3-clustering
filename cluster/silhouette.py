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

        # Find the number of clusters
        # Initialize an array to keep track of a (distance of each point to every other point in the same cluster)
        nclusters = np.amax(y) + 1
        nobs, nlabels = X.shape
        labels = np.arange(0, np.amax(y) + 1)
        a = np.zeros(nobs)

        # Loop over all clusters and calculate a by euclidean distance
        for cluster in np.unique(labels):            

            cluster_type = X[cluster == y]
            inds = np.where(cluster == y)
            dists = cdist(cluster_type, cluster_type, metric="euclidean")
            
            a[inds] = np.sum(dists, axis=1) / (dists.shape[0] - 1)

        b = np.zeros(nobs)

        # Loop over all clusters and calculate b by euclidean distance
        # b is the smallest mean distance for each point to another cluster
        for cluster in np.unique(labels):
            cluster_type_og = X[cluster == y]
            inds = np.where(cluster == y)
            other_labels = np.delete(y, inds)
            min_dists = np.full(X[cluster == y].shape[0], np.inf)
            
            for other_cluster in np.unique(other_labels):
                cluster_type = X[other_cluster == y]

                # Make sure that the orientation of the matrix is correct
                dists = cdist(cluster_type_og, cluster_type, metric="euclidean")
                avg_dists_per_cluster = np.mean(dists, axis=1)
                min_dists = np.minimum(avg_dists_per_cluster, min_dists)
            
            b[inds] = min_dists
            
        return (b - a) / (np.maximum(a, b))


