import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data = pd.read_csv(filename,
                       parse_dates=['Date'])
    data = data.dropna()
    data = data[data['Temp'] >= 0]
    daytime = data['Date'].dt.dayofyear
    daytime.name = 'DayOfYear'
    data = data.merge(daytime, left_index=True, right_index=True)

    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    temps = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    temps_israel = temps[temps['Country'] == 'Israel']
    temps_israel['Year'] = temps_israel['Year'].astype(str)
    fig_1 = px.scatter(temps_israel,
                       x='DayOfYear',
                       y='Temp',
                       color='Year',
                       title="Q3.2.2 - Temperatures according to Day of the year")
    fig_1.show()
    temps_israel['Year'] = temps_israel['Year'].astype(int)

    temps_std_by_month = temps_israel.groupby('Month').Temp.agg('std')
    fig_2 = px.bar(temps_std_by_month,
                   title="Q3.2.2 - Temperature's std by month")
    fig_2.show()


    # Question 3 - Exploring differences between countries
    temps_country_month = temps.groupby(['Country', 'Month']).Temp.agg(['mean', 'std']).reset_index()
    fig_3 = px.line(temps_country_month,
                    x='Month',
                    y='mean',
                    color='Country',
                    error_y='std',
                    title="Q3.2.3 - Average monthly temperature by countries")
    fig_3.show()

    # Question 4 - Fitting model for different values of `k`
    train_date_data, train_temp, test_date_data, test_temp = split_train_test(temps_israel['DayOfYear'].to_frame(), temps_israel['Temp'], 0.75)
    loss_by_k = np.zeros(10)

    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(train_date_data.to_numpy(), train_temp.to_numpy())
        loss_by_k[k-1] = pf._loss(test_date_data.to_numpy(), test_temp.to_numpy())
        print("loss for k = " + str(k) + " is " + "%.2f" % loss_by_k[k-1])

    fig_4 = go.Figure(go.Bar(
        x=np.arange(1, 11),
        y=loss_by_k))
    fig_4.update_layout(
        title="Q3.2.4 - Loss by Polynom Degree",
        xaxis_title="K",
        yaxis_title="Loss")
    fig_4.show()

    # Question 5 - Evaluating fitted model on different countries
    pf = PolynomialFitting(np.argmin(loss_by_k)+1)
    pf.fit(temps_israel['DayOfYear'].to_frame().to_numpy(), temps_israel['Temp'].to_numpy())
    other_countries = np.unique(temps[temps.Country != 'Israel'].Country.values)
    loss_by_country = np.zeros(other_countries.size)

    for c in range(other_countries.size):
        loss_by_country[c] = pf.loss(temps[temps.Country == other_countries[c]].DayOfYear.to_frame().to_numpy(),
                                     temps[temps.Country == other_countries[c]].Temp.to_numpy())

    fig_5 = go.Figure(go.Bar(
        x=other_countries,
        y=loss_by_country))
    fig_5.update_layout(
        title="Q3.2.5 - Loss by Country",
        xaxis_title="Country",
        yaxis_title="Loss")
    fig_5.show()


    c = 9