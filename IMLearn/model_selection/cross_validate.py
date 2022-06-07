from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    fold_indices = np.arange(X.shape[0])
    # np.random.shuffle(fold_indices)
    folds = np.array_split(fold_indices, cv)
    train_score = 0
    validation_score = 0

    for fold in folds:
        X_val = X[fold]
        y_val = y[fold]
        train_indices = np.ones(X.shape[0], bool)
        train_indices[fold] = False
        X_train = X[train_indices]
        y_train = y[train_indices]
        estimator.fit(X_train, y_train)

        y_train_pred = estimator.predict(X_train)
        y_val_pred = estimator.predict(X_val)
        train_score += scoring(y_train, y_train_pred)
        validation_score += scoring(y_val, y_val_pred)

    train_score /= cv
    validation_score /= cv

    return (train_score, validation_score)
