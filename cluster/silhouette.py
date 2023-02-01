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
        labels = np.arrange(0, np.amax(y))
        a = np.zeros(nclusters)

        for cluster in labels:
            cluster_type = X[cluster == y]
            dists = cdist(cluster_type, cluster_type, metric="euclidean")
            
            a[cluster] = np.sum(dists, axis=1)/(dists.shape[0] - 1)

        b = np.zeros(nclusters)

        for cluster in labels:
            #cluster_type = X[cluster != y]
            other_labels = np.delete(labels, cluster)
            min_dists = np.full(X[cluster == y].shape[0], np.inf)

            for other_cluster in other_labels:
                cluster_type = X[other_cluster == y]

                dists = cdist(cluster_type, cluster_type, metric="euclidean")
                avg_dists_per_cluster = np.mean(dists, axis=1)
                min_dists = np.minimum(avg_dists_per_cluster, min_dists)

            b[cluster] = min_dists

        return (b - a) / (np.maximum(a, b))


            #dists.ravel()[::dists.shape[1]+1] = dists.max()+1
            #test[dists.argmin(1)]

            #for obs in range(obs_in_cluster):
                # do not include pt itself
                #cdist(cluster_type[obs], cluster_type, metric="euclidean")

