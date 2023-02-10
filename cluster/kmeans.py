import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if k <= 0: raise Exception("k must be a positive integer") 
        if type(k) != int: raise TypeError("k must be a positive integer")

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        self.observations, self.features = mat.shape

        if self.observations < self.k: raise Exception("Cannot assign " + str(self.observations) + " observations to " + str(self.k) + " clusters")

        # Selection of uniformly random points as the initial centroids leads to poor clustering
        #self.centroids = mat[np.random.choice(self.observations, self.k, replace=False)]

        # Try k-means++ initialization, with private helper function _init_centroids
        self.centroids = self._init_centroids(mat)
        self.pred_labels = np.zeros((self.observations, 1))
     
        # Initialize and iterator and error for predictions
        i = 0
        error = np.inf

        # Loop until iterations exceeds mat_iter and the error exceeds tolerance
        while i < self.max_iter and error > self.tol:

            # Use provided functions to predict labels, assign the old centroids to the current centoirds
            # Get new centroids for the predicted labels, as well as the error
            self.pred_labels = self.predict(mat)
            self.old = self.centroids
            self.centroids = self.get_centroids(mat)
            error = self.get_error()
            i += 1


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        
        # Find the Euclidean distance between every point and its nearest centroid

        return np.argmin(cdist(mat, self.centroids, metric="euclidean"), axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """

        return np.sum(np.square(self.old - self.centroids))

    def get_centroids(self, mat:np.ndarray) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        fit_centroids = np.zeros((self.k, self.features))

        # Get centroids locations by taking the average of the predicted labels from cluster across all observations
        # Centroid will be the average of those points matching the corresponding label
        for cluster in range(self.k):            
            fit_centroids[cluster, :] = np.mean(mat[cluster == self.pred_labels, :], axis = 0)

        return fit_centroids

    def _init_centroids(self, mat:np.ndarray) -> np.ndarray:

        # Instead of random initialization, start with a random point as a centroid
        # The next centroid will be the point furthest away from the previous centroid until we reach k centroids
        # Modify mat to remove the previous centroid itself
        # We are replacing each random centroid with new centroids as we loop
        centroids = mat[np.random.choice(self.observations, self.k)]
        mod_mat = mat

        for centroid in range(self.k - 1):
            ind = np.where((mod_mat == centroids[centroid]).all(axis=1))
            mod_mat = np.delete(mod_mat, ind, axis=0)
            
            centroids[centroid + 1] = mat[np.argmax(np.sum(np.square(centroids[centroid] - mod_mat), axis=1))]

        return centroids

