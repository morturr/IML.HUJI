from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((len(self.classes_), X.shape[1]))
        self.vars_ = np.zeros((len(self.classes_), X.shape[1]))
        nk = np.zeros(len(self.classes_))
        for c in range(len(self.classes_)):
            nk[c] = len(y[y == self.classes_[c]])

        self.pi_ = nk / len(y)

        # mu computation
        for c in range(len(self.classes_)):
            X_c = X[(y == self.classes_[c])]
            self.mu_[c, :] = np.mean(X_c, axis=0)
            w = X_c - self.mu_[c]
            self.vars_[c, :] = np.sum(np.power(w, 2), axis=0) / (nk[c] - 1)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        likelihoods = self.likelihood(X)
        posterior = likelihoods + np.log(self.pi_)
        responses = np.argmax(posterior, axis=1)
        return responses

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        likelihoods = np.zeros((X.shape[0], len(self.classes_)))

        for c in range(len(self.classes_)):
            cov_inv = np.linalg.inv(np.diagflat(self.vars_[c, :]))
            log_det_cov_inv = np.log(np.linalg.det(cov_inv))
            d = X[:, np.newaxis, :] - self.mu_[c, :]
            likelihoods[:, c] = .5 * (log_det_cov_inv - np.sum(d.dot(cov_inv) * d, axis=2).flatten())

        return likelihoods

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        from ...metrics import misclassification_error
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
