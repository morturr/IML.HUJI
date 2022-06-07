from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn
from scipy.constants import alpha
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.random.uniform(-1.2, 2, n_samples)
    eps = np.random.randn(n_samples) * noise
    f_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    y = f_x + eps

    train_x, train_y, test_x, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3)
    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    fig1 = go.Figure([go.Scatter(x=x, y=f_x, mode='markers', name='True values'),
                      go.Scatter(x=train_x.squeeze(), y=train_y, mode='markers', name='Train values'),
                      go.Scatter(x=test_x.squeeze(), y=test_y, mode='markers', name='Test values')],
                     layout=go.Layout(title=f'Noiseless and Noisy values of y,'
                                            f' Noise level={noise}, m={n_samples}'))
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errs = []
    validation_errs = []
    for k in range(11):
        pf = PolynomialFitting(k)
        tr_err, val_err = cross_validate(pf, train_x, train_y, mean_square_error)
        train_errs.append(tr_err)
        validation_errs.append(val_err)

    fig2 = go.Figure([go.Scatter(x=np.arange(11), y=train_errs, name='Train errors', mode='lines+markers'),
                      go.Scatter(x=np.arange(11), y=validation_errs, name='Validation errors', mode='lines+markers')],
                      layout=go.Layout(title=f'Train and Validation error according to different values of k,'
                                             f' Noise level={noise}, m={n_samples}',
                                       xaxis_title='k'))
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = np.argmin(np.array(validation_errs))
    pf = PolynomialFitting(k_star)
    pf.fit(train_x, train_y)
    k_star_test_err = mean_square_error(test_y, pf.predict(test_x))
    print(f'best k is {k_star}, test error is {round(k_star_test_err, 2)}')

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x = X[:n_samples, :]
    train_y = y[:n_samples]
    test_x = X[n_samples:, :]
    test_y = y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_errs = []
    ridge_val_errs = []
    lasso_train_errs = []
    lasso_val_errs = []
    my_range = np.linspace(0.001, 2, n_evaluations)

    for lam in my_range:
        ridge = RidgeRegression(lam)
        train_err, val_err = cross_validate(ridge, train_x, train_y, mean_square_error)
        ridge_train_errs.append(train_err)
        ridge_val_errs.append(val_err)

        lasso = sklearn.linear_model.Lasso(alpha=lam)
        train_err, val_err = cross_validate(lasso, train_x, train_y, mean_square_error)
        lasso_train_errs.append(train_err)
        lasso_val_errs.append(val_err)

    fig7 = go.Figure([go.Scatter(x=my_range, y=ridge_train_errs, name='Ridge Train error'),
                      go.Scatter(x=my_range, y=ridge_val_errs, name='Ridge Validation error'),
                      go.Scatter(x=my_range, y=lasso_train_errs, name='Lasso Train error'),
                      go.Scatter(x=my_range, y=lasso_val_errs, name='Lasso Validation error')])

    fig7.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = my_range[np.argmin(np.array(ridge_val_errs))]
    best_lam_lasso = my_range[np.argmin(np.array(lasso_val_errs))]
    print(f'Best regularization parameter for Ridge is {best_lam_ridge}')
    print(f'Best regularization parameter for Lasso is {best_lam_lasso}')

    ridge = RidgeRegression(best_lam_ridge)
    ridge.fit(train_x, train_y)
    ridge_loss = ridge.loss(test_x, test_y)

    lasso = sklearn.linear_model.Lasso(alpha=best_lam_lasso)
    lasso.fit(train_x, train_y)
    lasso_loss = mean_square_error(test_y, lasso.predict(test_x))

    lr = LinearRegression()
    lr.fit(train_x, train_y)
    lr_loss = lr.loss(test_x, test_y)

    print(f'Ridge error = {ridge_loss}')
    print(f'Lasso error = {lasso_loss}')
    print(f'Least Squares error = {lr_loss}')


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()