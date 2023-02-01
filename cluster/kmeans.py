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

        if k <= 0: raise Exception("k must be a positive integer") #TypeError

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

        self.mat = mat
        self.observations, self.features = self.mat.shape

        if self.observations < self.k: raise Exception("Cannot assign " + str(self.observations) + " observations to " + str(self.k) + " clusters")

        self.centroids = mat[np.random.choice(self.observations, self.k, replace=False)]
        self.pred_labels = np.zeros((self.observations, 1))
     
        i = 0
        error = np.inf

        while i < self.max_iter and error > self.tol:
            self.pred_labels = self.predict(self.mat)
            print("pred labels: ", self.pred_labels)
            self.centroids = self.get_centroids()
            if i == 1: error = self.get_error()
            else: error = error - self.get_error()
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
        
        return np.argmin(cdist(self.mat, self.centroids, metric="euclidean"), axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        mse = np.zeros(self.k)

        for cluster in range(self.k):
            mse[cluster] = np.sum(np.square(self.mat[cluster == self.pred_labels] - self.centroids[cluster]))
        return np.sum(mse)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        fit_centroids = np.zeros((self.k, self.features))

        for cluster in range(self.k):
            #print("centroids fit: ", self.mat[cluster == self.pred_labels, :])
            #break
            fit_centroids[cluster, :] = np.mean(self.mat[cluster == self.pred_labels, :], axis = 0)

        return fit_centroids


