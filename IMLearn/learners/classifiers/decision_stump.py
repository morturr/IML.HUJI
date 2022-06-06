from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error

class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # thr_miss_arr = np.zeros((X.shape[1], 2, 2))
        # for feature in range(X.shape[1]):
        #     for sign in([-1, 1]):
        #         val = self._find_threshold(X[:, feature], y, sign)
        #         thr_miss_arr[feature, 0, (int)((sign + 1) / 2)] = val[0]
        #         thr_miss_arr[feature, 1, (int)((sign + 1) / 2)] = val [1]
        #
        # idx = np.argmin(thr_miss_arr, axis=1)
        # self.threshold_ = idx[0]
        # self.j_ = idx[1]
        # self.sign_ = idx[2] * 2 - 1

        thr_miss_arr_plus = np.zeros((X.shape[1], 2))
        thr_miss_arr_minus = np.zeros((X.shape[1], 2))
        for j in range(X.shape[1]):
            val = self._find_threshold(X[:, j], y, -1)
            thr_miss_arr_minus[j, 0], thr_miss_arr_minus[j, 1] = val[0], val[1]
            val = self._find_threshold(X[:, j], y, 1)
            thr_miss_arr_plus[j, 0], thr_miss_arr_plus[j, 1] = val[0], val[1]

        minus_err, minus_idx = np.min(thr_miss_arr_minus[:, 1]), np.argmin(thr_miss_arr_minus[:, 1])
        plus_err, plus_idx = np.min(thr_miss_arr_plus[:, 1]), np.argmin(thr_miss_arr_plus[:, 1])

        if minus_err <= plus_err:
            self.sign_ = -1
            self.j_ = minus_idx
            self.threshold_ = thr_miss_arr_minus[minus_idx, 0]

        else:
            self.sign_ = 1
            self.j_ = plus_idx
            self.threshold_ = thr_miss_arr_plus[plus_idx, 0]

        a = 9

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

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        responses = np.zeros(X.shape[0])
        responses[X[:, self.j_] >= self.threshold_] = self.sign_
        responses[X[:, self.j_] < self.threshold_] = -self.sign_

        return responses

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        val_lab = np.column_stack((values, labels))
        val_lab = val_lab[val_lab[:, 0].argsort()]
        pred_lab = np.zeros(labels.shape[0])
        min_thr_err = 1
        best_thr = 0

        for thr in val_lab[:, 0]:
            pred_lab[val_lab[:, 0] >= thr] = sign
            pred_lab[val_lab[:, 0] < thr] = -sign
            misses = np.sign(pred_lab) != np.sign(val_lab[:, 1])
            thr_err = np.sum(np.abs(val_lab[:, 1][misses])) / np.sum(np.abs(val_lab[:, 1]))
            if thr_err < min_thr_err:
                min_thr_err = thr_err
                best_thr = thr

        return (best_thr, min_thr_err)


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
        y_pred = self.predict(X)
        misses = np.sign(y_pred) != np.sign(y)
        loss = np.sum(np.abs(y[misses]))
        return loss