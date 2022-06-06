import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    if type(noise) is not list:
        noise = [noise]

    for n in noise:
        (train_X, train_y), (test_X, test_y) = generate_data(train_size, n), generate_data(test_size, n)

        # Question 1: Train- and test errors of AdaBoost in noiseless case
        def ds_generator():
            return DecisionStump()
        adaboost = AdaBoost(wl=ds_generator, iterations=n_learners)
        adaboost.fit(train_X, train_y)
        losses_test = []
        losses_train = []

        for t in range(n_learners):
            losses_test.append(adaboost.partial_loss(test_X, test_y, t+1))
            losses_train.append(adaboost.partial_loss(train_X, train_y, t + 1))

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=losses_train, name='Train loss'))
        fig1.add_trace(go.Scatter(y=losses_test, name='Test loss'))
        fig1.update_layout(title=rf"$\textbf{{AdaBoost loss by # learners, noise level = {n}}}$",
                           xaxis_title="Number of learners",
                           yaxis_title="Missclassification error")
        fig1.show()

        lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
        if n == 0: # no noise
            # Question 2: Plotting decision surfaces
            T = [5, 50, 100, 250]
            fig2 = make_subplots(rows=2, cols=2,
                                subplot_titles=[rf"$\textbf{{Decision Boundary of {t} learners}}$" for t in T],
                                horizontal_spacing=0.01, vertical_spacing=.03)
            for i, t in enumerate(T):
                fig2.add_traces([decision_surface(adaboost.partial_predict, t, lims[0], lims[1], showscale=False),
                                go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                           marker=dict(color=train_y, colorscale=custom,
                                                       line=dict(color="black", width=1))),
                                go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                           marker=dict(color=test_y, colorscale=custom,
                                                       line=dict(color="black", width=1)))],
                               rows=(i // 2) + 1, cols=(i % 2) + 1)
            fig2.update_layout(title=rf"$\textbf{{Decision Boundaries according to different # of learners}}$",
                              margin=dict(t=100)) \
                .update_xaxes(visible=False).update_yaxes(visible=False)
            fig2.show()

            # Question 3: Decision surface of best performing ensemble
            losses_by_t = []
            for i, t in enumerate(T):
                losses_by_t.append((t, adaboost.partial_loss(test_X, test_y, t)))
            t_loss_tuple = min(losses_by_t, key = lambda p: p[1])
            pred_y_t = adaboost.partial_predict(test_X, t_loss_tuple[0])
            fig3 = go.Figure()
            fig3.add_traces([decision_surface(adaboost.partial_predict,t_loss_tuple[0], lims[0], lims[1], showscale=False),
                            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=train_y, colorscale=custom,
                                                   line=dict(color="black", width=1))),
                            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=test_y, colorscale=custom,
                                                   line=dict(color="black", width=1)))])
            from IMLearn.metrics import accuracy
            fig3.update_layout(title=rf"$\textbf{{Decision Boundary of {t_loss_tuple[0]} learners, accuracy = {accuracy(pred_y_t, test_y)}}}$",
                              margin=dict(t=100)) \
                .update_xaxes(visible=False).update_yaxes(visible=False)
            fig3.show()

        # Question 4: Decision surface with weighted samples
        fig4 = go.Figure().add_traces([go.Scatter(x=train_X[:,0], y=train_X[:, 1],  mode='markers',
                                    marker=dict(size=(10 * adaboost.D_ / np.max(adaboost.D_)), symbol=(train_y+1),
                                                color=(train_y+1), colorscale=custom,
                                                line=dict(color="black", width=1))),
                         decision_surface(adaboost.partial_predict, n_learners, lims[0], lims[1], showscale=False)
                         ])
        fig4.update_layout(title=rf"$\textbf{{Train data points with weighted size, noise level = {n}}}$") \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig4.show()



if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=[0, 0.4], n_learners=250)
