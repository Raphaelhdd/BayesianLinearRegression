import os

import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    """
    Create a function that calculates the polynomial basis functions up to (and including) a degree
    :param degree: the maximal degree of the polynomial basis functions
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             polynomial basis functions, a numpy array of shape [N, degree+1]
    """

    def pbf(x: np.ndarray):
        # Create a matrix of one because the first column of the index 0 is for the bias
        design_matrix = np.ones((len(x), degree + 1))
        for i in range(1, degree + 1):
            # Fill the column i with (x/i)**i
            design_matrix[:, i] = (x / i) ** i

        return design_matrix

    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    """
    Create a function that calculates the fourier basis functions up to a certain frequency
    :param num_freqs: the number of frequencies to use
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Fourier basis functions, a numpy array of shape [N, 2*num_freqs + 1]
    """

    def fbf(x: np.ndarray):
        design_matrix = np.zeros((len(x), 2 * num_freqs + 1))
        for k in range(1, num_freqs + 1):
            design_matrix[:, 2 * k - 1] = np.cos(2 * np.pi * k * x / 24)
            design_matrix[:, 2 * k] = np.sin(2 * np.pi * k * x / 24)

        # Add the bias term
        design_matrix[:, 0] = 1

        return design_matrix

    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    """
    Create a function that calculates the cubic regression spline basis functions around a set of knots
    :param knots: an array of knots that should be used by the spline
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             cubic regression spline basis functions, a numpy array of shape [N, len(knots)+4]
    """

    def csbf(x: np.ndarray):
        # Creates the first 4th colums to 1,t,t^2 and t^3 and the last for the knots
        design_matrix = np.ones((len(x), len(knots) + 4))
        for row in range(len(x)):
            for col in range(1, 4):
                design_matrix[row][col] = (x[row]) ** col
        for row in range(len(x)):
            for index_knot in range(len(knots)):
                if x[row] - knots[index_knot] >= 0:
                    design_matrix[row][4 + index_knot] = (x[row] - knots[index_knot]) ** 3
                else:
                    design_matrix[row][4 + index_knot] = 0
        return design_matrix

    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    """
    Learn a Gaussian prior using historic data
    :param hours: an array of vectors to be used as the 'X' data
    :param temps: a matrix of average daily temperatures in November, as loaded from 'jerus_daytemps.npy', with shape
                  [# years, # hours]
    :param basis_func: a function that returns the design matrix of the basis functions to be used
    :return: the mean and covariance of the learned covariance - the mean is an array with length dim while the
             covariance is a matrix with shape [dim, dim], where dim is the number of basis functions used
    """
    thetas = []
    # iterate over all past years
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.thetas)  # add thetas that we learned

    thetas = np.array(thetas)  # create arrays of thetas

    # take mean over parameters learned each year for the mean of the prior
    mu = np.mean(thetas, axis=0)
    # calculate empirical covariance over parameters learned each year for the covariance of the prior
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        """
        Initializes a Bayesian linear regression model
        :param theta_mean:          the mean of the prior
        :param theta_cov:           the covariance of the prior
        :param sig:                 the signal noise to use when fitting the model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.inv_M = None
        self.post_mean = None
        self.post_cov = None
        self.theta_mean = theta_mean  # mu
        self.theta_cov = theta_cov  # cov
        self.sig = sig  # noise
        self.basis_functions = basis_functions  # h(.)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        """
        Find the model's posterior using the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # We will use the equivalence form from the 4 in the tirgul
        # As always, h(.) * X
        H = self.basis_functions(X)
        self.inv_M = np.linalg.inv(self.sig * np.eye(len(X)) + H @ self.theta_cov @ H.T)
        self.post_mean = self.theta_mean + self.theta_cov @ H.T @ self.inv_M @ (y - H @ self.theta_mean)
        self.post_cov = self.theta_cov - self.theta_cov @ H.T @ self.inv_M @ H @ self.theta_cov
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model using MMSE
        :param X: the samples to predict
        :return: the predictions for X
        """
        # return H.theta
        return self.basis_functions(X) @ self.post_mean

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Find the model's posterior and return the predicted values for X using MMSE
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the model's standard deviation around the mean prediction for the values of X
        :param X: the samples around which to calculate the standard deviation
        :return: a numpy array with the standard deviations (same shape as X)
        """
        H = self.basis_functions(X)
        return np.sqrt(np.sum(np.dot(H, self.post_cov) * H, axis=1))

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model and sampling from the posterior
        :param X: the samples to predict
        :return: the predictions for X
        """
        H = self.basis_functions(X)
        samples = np.random.multivariate_normal(self.post_mean, self.post_cov, size=len(X))
        return np.dot(H, samples.T)


class LinearRegression:

    def __init__(self, basis_functions: Callable):
        """
        Initializes a linear regression model
        :param basis_functions:     a function that receives data points as inputs and returns a design matrix
        """
        self.basis_functions = basis_functions  # function h(.)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the model to the training data X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the fitted model
        """
        # from the tirgul θML = (H.T @ H)^-1 @ H.T @ y
        H = self.basis_functions(X)
        self.thetas = np.linalg.inv(H.T @ H) @ H.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the regression values of X with the trained model
        :param X: the samples to predict
        :return: the predictions for X
        """
        # y = H @ θ
        return self.basis_functions(X) @ self.thetas

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the model and return the predicted values for X
        :param X: the training data
        :param y: the true regression values for the samples X
        :return: the predictions of the model for the samples X
        """
        self.fit(X, y)
        return self.predict(X)


def plot_prior_graph(pbf, x, mu, cov, title, title_save):
    """
    Plots the prior distribution of the Bayesian linear regression model.

    :param pbf: Function that returns the design matrix of the basis functions.
    :param x: Array of input values.
    :param mu: Mean of the prior distribution.
    :param cov: Covariance matrix of the prior distribution.
    :param title: Title of the plot.
    :param title_save: Title to save the plot as an image.
    """
    H = pbf(x)
    mean_func = H @ mu
    std_dev = np.sqrt(np.diag(H @ cov @ H.T))
    upper_bound = mean_func + std_dev
    lower_bound = mean_func - std_dev
    plt.figure()
    plt.fill_between(x, upper_bound, lower_bound, color='blue', alpha=.5, label='Confidence Interval')
    for i in range(5):
        sampled_coefficients = np.random.multivariate_normal(mu, cov)
        sampled_function = np.dot(H, sampled_coefficients)
        plt.plot(x, sampled_function)
    plt.plot(x, mean_func, color='black', lw=3, label='prior mean')
    plt.xlabel('hour')
    plt.ylabel('temperature')
    plt.title(title)
    plt.legend()
    plt.xlim([0, 23.5])
    # plt.savefig(f'../Bayesian/plots/{title_save}')
    plt.show()


def plot_posterior_graph(blr, train, train_hours, test, test_hours, x, title, title_save,for_print):
    """
    Plots the posterior distribution of the Bayesian linear regression model.

    :param blr: BayesianLinearRegression object.
    :param train: Training data.
    :param train_hours: Hours corresponding to the training data.
    :param test: Test data.
    :param test_hours: Hours corresponding to the test data.
    :param x: Array of input values.
    :param title: Title of the plot.
    :param title_save: Title to save the plot as an image.
    :param for_print: String for printing purposes.
    """
    blr.fit(train_hours, train)
    predict_test = blr.predict(test_hours)
    predict_train = blr.predict(train_hours)
    predict_combined = np.concatenate((predict_train, predict_test))
    combined_hours = np.concatenate((train_hours, test_hours))
    std_dev_combined = blr.predict_std(combined_hours)
    upper_bound_combined = predict_combined + std_dev_combined
    lower_bound_combined = predict_combined - std_dev_combined
    plt.plot(combined_hours, predict_combined, color='black', lw=3, label='MMSE')
    plt.scatter(test_hours, test, color='orange', label='Test points')
    plt.scatter(train_hours, train, color='blue', label='Train points')
    plt.fill_between(combined_hours, lower_bound_combined, upper_bound_combined, color='blue', alpha=0.5,
                     label='Confidence Interval')
    posterior_samples = blr.posterior_sample(x)
    for i in range(5):
        plt.plot(x, posterior_samples[:, i])
    print(f'Average squared error {for_print} is {np.mean((test - predict_test) ** 2):.2f}')
    plt.xlabel('Hour')
    plt.ylabel('Temperature')
    plt.title(title + f'E={np.mean((test - predict_test) ** 2):.2f}')
    plt.legend()
    plt.show()
    plt.xlim([0, 23.5])
    # plt.savefig(f'../Bayesian/plots/{title_save}')


def main():
    # os.makedirs('../Bayesian/plots', exist_ok=True)
    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')  # Here 48 points of temperature
    nov16_hours = np.arange(0, 24, .5)  # here 0 to 23.5 (48 points) that are the time
    train = nov16[:len(nov16) // 2]  # The first 24 points of temperature
    train_hours = nov16_hours[:len(nov16) // 2]  # from time 0 to 11.5
    test = nov16[len(nov16) // 2:]  # The last 24 points of temperature
    test_hours = nov16_hours[len(nov16) // 2:]  # from time 12 to 23.5

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)

        # print average squared error performance
        print(f'Average squared error with LR and d={d} is {np.mean((test - ln.predict(test_hours)) ** 2):.2f}')

        # plot graphs for linear regression part
        plt.scatter(train_hours, train, color='blue', label='train')
        plt.scatter(test_hours, test, color='orange', label='test')
        plt.plot(nov16_hours, ln.predict(nov16_hours), color='black', label='prediction')
        plt.legend()
        plt.xlabel('hour')
        plt.ylabel('temperature')
        plt.title(f'Classical Linear Regression with $d$={d}, E={np.mean((test - ln.predict(test_hours)) ** 2):.2f}')
        # plt.savefig(f'../Bayesian/plots/classic_linear_regression_degree_{d}.png')
        plt.show()

    # ----------------------------------------- Bayesian Linear Regression

    # load the historic data
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    # setup the model parameters
    sigma = 0.25
    degrees = [3, 7]  # polynomial basis functions degrees

    # frequencies for Fourier basis
    freqs = [1, 2, 3]
    #
    # sets of knots K_1, K_2 and K_3 for the regression splines
    knots = [np.array([12]),
             np.array([8, 16]),
             np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)

        blr = BayesianLinearRegression(mu, cov, sigma, pbf)

        # plot prior graphs
        plot_prior_graph(pbf, x, mu, cov, f'Prior Polynomial BLR with degree d={deg}),',
                         f'prior_polynomial_blr_{deg}.png')

        # plot posterior graphs
        plot_posterior_graph(blr, train, train_hours, test, test_hours, x, f'Polynomial BLR with degree $d={deg}$, ',
                             f'polynomial_blr_{deg}.png',f'with BLR and d={deg} ')

    # ---------------------- Gaussian basis functions
    for ind, K in enumerate(freqs):
        rbf = fourier_basis_functions(K)
        mu, cov = learn_prior(hours, temps, rbf)

        blr = BayesianLinearRegression(mu, cov, sigma, rbf)

        # plot prior graphs
        plot_prior_graph(rbf, x, mu, cov, f'Prior Fourier BLR with degree K={K}),',
                         f'prior_fourier_blr_{K}.png')

        # plot posterior graphs
        plot_posterior_graph(blr, train, train_hours, test, test_hours, x, f'Fourier BLR with degree $K={K}$, ',
                             f'fourier_blr_{K}.png',f'with Fourier BLR and K={K} ')

    # ---------------------- cubic regression splines
    k_list_title = {
        0: [12],
        1: [8, 16],
        2: [6, 12, 18]
    }
    for ind, k in enumerate(knots):
        spline = spline_basis_functions(k)
        mu, cov = learn_prior(hours, temps, spline)

        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        # plot prior graphs
        plot_prior_graph(spline, x, mu, cov, f'Spline Prior BLR with set $K={k_list_title[ind]}$, ',
                         f'prior_spline_blr_{k_list_title[ind]}.png')
        # plot posterior graphs
        plot_posterior_graph(blr, train, train_hours, test, test_hours, x,
                             f'Spline Posterior BLR with set $K={k_list_title[ind]}$, ',
                             f'posterior_spline_blr_{k_list_title[ind]}.png',f'with Spline BLR and set K={k_list_title[ind]} ')


if __name__ == '__main__':
    main()
