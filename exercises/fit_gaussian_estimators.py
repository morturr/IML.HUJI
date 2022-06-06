import plotly

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    n_samples = 1000
    samples_arr = np.random.normal(loc=mu, scale=var, size=n_samples)
    ug = UnivariateGaussian()
    ug.fit((samples_arr))

    print("(" + str(ug.mu_) + ", " + str(ug.var_) + ")")


    # Question 2 - Empirically showing sample mean is consistent
    sample_sizes = np.arange(10, 1001, 10)
    dist_arr = np.zeros(sample_sizes.shape)

    for k in sample_sizes:
        index = (int)((k/10) -1)
        dist_arr[index] = np.abs(ug.fit(samples_arr[1:k]).mu_ - 10)

    q2_fig = go.Figure(data = go.Scatter(x=sample_sizes, y=dist_arr))
    q2_fig.update_layout(
        title="Question 3.1.2 - Samples from ~N(10,1) and difference from estimated mean",
        xaxis_title="Sample Size",
        yaxis_title="Difference between estimated and original mean"
    )
    q2_fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    ug.fit(samples_arr)
    pdfs = ug.pdf(samples_arr)
    q3_fig = go.Figure(data=go.Scatter(x=samples_arr, y=pdfs, mode='markers'))
    q3_fig.update_layout(
        title="Question 3.1.3 - Samples from ~N(10,1) and PDFs",
        xaxis_title="Sample Value",
        yaxis_title="Sample PDF value"
    )
    q3_fig.show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array(
        [[1, 0.2, 0, 0.5],
         [0.2, 2, 0, 0],
         [0, 0, 1, 0],
         [0.5, 0, 0, 1]])
    n_samples = 1000

    samples_arr = np.random.multivariate_normal(mean=mu, cov=cov, size=n_samples)
    mg = MultivariateGaussian()
    mg.fit(samples_arr)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    log_likelihood = np.zeros((len(f1), len(f3)))

    for i in range(len(f1)):
        for j in range(len(f3)):
            mu_ij = [f1[i], 0, f3[j], 0]
            log_likelihood[i, j] = mg.log_likelihood(mu_ij, cov, samples_arr)
    q6_fig = go.Figure(go.Heatmap(x=f1, y=f3, z=log_likelihood))
    q6_fig.update_layout(
            title="Question 3.2.5 - Log-Likelihood for different expectation vectors",
            xaxis_title="f1",
            yaxis_title="f3",
            autosize=False,
            width=800,
            height=800)
    q6_fig.show()

    # Question 6 - Maximum likelihood
    max_f1 = f1[np.argmax(log_likelihood, axis=0)[0]]
    max_f3 = f3[np.argmax(log_likelihood, axis=1)[0]]
    print("argmax_f1 = " + str("%.3f" % max_f1) + ", argmax_f3 = " + str("%.3f" % max_f3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
