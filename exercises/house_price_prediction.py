
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors.linear_regression import LinearRegression
import os

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename)
    #data = data.dropna().drop(['id', 'date', 'condition', 'lat', 'long',
      #                         'sqft_lot15', 'sqft_lot', 'yr_built'], axis=1)

    data = data.dropna().drop(['id', 'date'], axis=1)

    data = data[(data['price'] > 0) & \
                (data['bedrooms'] > 0) & \
                (data['bathrooms'] > 0) & \
                (data['sqft_living'] > 30) & \
                (data['floors'] > 0)]

    data['age'] = (2015 - data['yr_built'])
    data.age = [
        0 if i < 1
        else 1 if i < 5
        else 2 if i < 10
        else 3 if i < 25
        else 4 if i < 50
        else 5 if i < 75
        else 6 if i < 100
        else 7
        for i in data.age]

    data['renov_age'] = (2015 - data['yr_renovated'])
    data.renov_age = [
        0 if i < 1
        else 1 if i < 5
        else 2 if i < 10
        else 3 if i < 25
        else 4 if i < 50
        else 5 if i < 75
        else 6 if i < 100
        else 7
        for i in data.renov_age]

    data['new_age'] = data['age'] + data['renov_age']

    #data['rooms_per_ft'] = data['sqft_living'] / data['bedrooms']
    #data['bath_per_room'] = data['bathrooms'] / data['bedrooms']
    data['sqft_living_vs_neigbors'] = data['sqft_living'] / data['sqft_living15']
    data['sqft_lot_vs_neigbors'] = data['sqft_lot'] / data['sqft_lot15']
    data['living_vs_lot'] = data['sqft_living'] / data['sqft_lot']
    data = data.drop(['age', 'renov_age'], axis=1)

    return data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not output_path is ".":
        dir_create = False
        i = 1
        while not dir_create:
            if os.path.exists(output_path + str(i)):
                i = i + 1
            else:
                os.mkdir(output_path + str(i))
                dir_create = True
                output_path = output_path + str(i)

    n_features = X.shape[1]
    covs = np.zeros((n_features, 1))
    y_std = y.std()
    y_name = y.name

    for i in range(n_features):
        covs[i] = X.iloc[:, i].cov(y) / (X.iloc[:, i].std() * y_std)
        pearson_fig = go.Figure(data=go.Scatter(x=X.iloc[:, i], y=y, mode='markers'))
        pearson_fig.update_layout(
            title="Pearson Correlation = " + str(covs[i]),
            xaxis_title="Feature - " + X.iloc[:, i].name,
            yaxis_title=y_name
        )

        image_loc_str = output_path + "/" + X.iloc[:, i].name + ".png"
        pearson_fig.write_image(image_loc_str)
        print("Pearson_Corr of " + X.iloc[:, i].name + " = " + str(covs[i]))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    filename = "../datasets/house_prices.csv"
    data = load_data(filename)

    # Question 2 - Feature evaluation with respect to response
    obs = data.drop(['price'], axis=1)
    prices = data['price']
    fe_path = "./pearson_correlation_"
    feature_evaluation(obs, prices, fe_path)

    # Question 3 - Split samples into training- and testing sets.
    train_house_data, train_prices, test_house_data, test_prices = split_train_test(obs, prices, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    lr = LinearRegression()
    sample_count = 100 - 10 + 1
    sample_means = np.zeros(sample_count)
    sample_stds = np.zeros(sample_count)

    for p in range(10, sample_count + 10):
        mean_loss = np.zeros(10)
        for i in range(10):
            train_p = pd.DataFrame.sample(train_house_data.merge(train_prices, left_index=True, right_index=True), frac=(p/100))
            lr.fit(train_p.drop(['price'], axis=1).to_numpy(), train_p['price'].to_numpy())
            mean_loss[i] = lr.loss(test_house_data.to_numpy(), test_prices.to_numpy())
            #y_pred = lr.predict(test_house_data.to_numpy())
            #mean_loss[i] = mean_square_error(test_prices.to_numpy(), y_pred)
        sample_means[p-10] = np.mean(mean_loss)
        sample_stds[p-10] = np.std(mean_loss)

    q4_fig = go.Figure([go.Scatter
                        (x=np.arange(10, 101),
                         y=sample_means,
                         line=dict(color='rgb(31, 119, 180)')),
                        go.Scatter(
                            x=np.arange(10, 101),
                            y=sample_means+2*sample_stds,
                            mode='lines',
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            hoverinfo="skip",
                            showlegend=False
                        ),
                        go.Scatter(
                            x=np.arange(10, 101),
                            y=sample_means-2*sample_stds,
                            marker=dict(color="#444"),
                            line=dict(width=0),
                            mode='lines',
                            fillcolor='rgba(68, 68, 68, 0.3)',
                            fill='tonexty',
                            showlegend=False
                        )
                        ])
    q4_fig.update_layout(
        title="Mean Loss as function of Sample Size",
        xaxis_title="% Sample Size",
        yaxis_title="Mean Loss"
    )

    q4_fig.show()






