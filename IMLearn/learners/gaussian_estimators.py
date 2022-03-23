from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        ddof_var = 1  # for unbiased estimator
        if (self.biased_):
            ddof_var = 0

        self.mu_ = np.mean(X)
        avg = np.sum(X) / len(X)
        self.var_ = np.var(X, ddof=ddof_var)
        self.fitted_ = True

        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        pdfs = np.exp(np.power(X - self.mu_, 2) / (-2 * self.var_)) / np.sqrt(2 * np.pi * self.var_)
        return pdfs

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        n_samples = len(X)
        log_likelihood_val = -1 * n_samples * np.log(np.sqrt(2 * np.pi * np.power(sigma, 2))) \
                             - np.sum(np.power(X - mu, 2)) / (2 * np.power(sigma, 2))

        return log_likelihood_val


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        n_samples = len(X[:, 0])
        n_features = len(X[0, :])
        self.mu_ = np.zeros(n_features)
        self.cov_ = np.zeros((n_features, n_features))
        diffs = np.zeros(np.shape(X))

        for i in range(n_features):
            self.mu_[i] = np.mean(X[:, i])

        for i in range(n_features):
            diffs[:, i] = X[:, i] - self.mu_[i]

        self.cov_ = np.cov(m=diffs, rowvar=False)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        n_samples = len(X[:, 0])
        n_features = len(X[0, :])
        pdfs = np.zeros(n_samples)
        diffs = np.zeros(np.shape(X))

        for i in range(n_features):
            diffs[i, :] = X[i, :] - self.mu_[i]

        pdfs = np.exp(-1/2 * diffs @ np.linalg.inv(self.cov_)*diffs) / \
            np.sqrt(np.power(2 * np.pi, n_features) * np.linalg.slogdet(self.cov_)[1])

        return pdfs


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        n_features = len(mu)
        n_samples = len(X[:, 0])
        diffs = np.zeros(np.shape(X))

        for i in range(n_features):
            diffs[:, i] = X[:, i] - mu[i]

        log_likelihood_val = -n_samples / 2 * \
                             (n_features * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1]) \
                             - 1 / 2 * np.sum(diffs @ np.linalg.inv(cov) * diffs)

        return log_likelihood_val
