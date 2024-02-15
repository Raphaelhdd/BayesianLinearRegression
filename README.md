# Bayesian Linear Regression for Temperature Prediction

This repository contains Python code for predicting temperatures using Bayesian linear regression models. 
The models are trained and tested on temperature data for November 16, 2024, with historical temperature data for 
Jerusalem used for training. The repository also includes scripts for generating various types of basis functions, 
learning priors from historical data, and visualizing the prior and posterior distributions.


## Problem Statement

We aim to predict temperatures for the second half of November 16, 2024, by fitting linear regression models 
with various basis functions. The linear regression model is given by:


$$
y_i(t) = \theta^T h(t) + \eta_i
$$



where \(y_i(t)\) represents the temperature at time \(t\), \(h(t)\) is a set of basis functions, \(\theta\) 
is the weight vector, and \(\eta_i\) is the noise term.

## Implementation

### 1. Classical Linear Regression

Implemented a class for classical linear regression, capable of fitting a linear regression model to a given set of 
data points and predicting new values.

### 2. Bayesian Linear Regression

Developed a class for Bayesian linear regression, which accepts a Gaussian prior and a set of basis functions. 
It calculates the posterior given a set of data points and predicts new values.

### Basis Functions

#### 2.1 Polynomial Basis Functions

Polynomial basis functions are implemented, which form a polynomial of degree \(d\).
For numerical stability, modified basis functions are used, where large numbers are divided by the degree.

#### 2.2 Fourier Basis Functions

Fourier basis functions are defined to capture the oscillatory nature of temperature variations throughout the day.

#### 2.3 Cubic Regression Splines

Cubic regression splines are implemented as piecewise polynomial functions with continuous second derivatives.


## Tasks

### Polynomial Basis Functions

- Fit linear regression models with polynomial basis functions of degrees 3 and 7.
- Learn a Gaussian prior over \(\theta\) using historical temperature data.
- Plot the mean functions described by the prior along with confidence intervals.
- Fit Bayesian linear regression models to predict temperatures for the second half of November 16, 2024.

### Fourier Basis Functions

- Implement Fourier basis functions for different numbers of frequencies.
- Fit Bayesian linear regression models using Fourier basis functions.
- Evaluate the performance and compare with polynomial basis functions.

### Cubic Regression Splines

- Implement cubic regression spline basis functions with different sets of knots.
- Fit Bayesian linear regression models using cubic regression splines.
- Compare the performance with polynomial and Fourier basis functions.

## Results

- Evaluate the average squared error for each method and compare the performance.
- Visualize the predictions and model distributions to gain insights into the temperature prediction task.

## Files

- `nov162024.npy`: Contains temperature data for November 16, 2024. It includes hourly temperature measurements.
- `jerus_daytemps.npy`: Historical temperature data for Jerusalem used for training the regression models.
- `ex2.py`: Python script implementing Bayesian linear regression models and visualization code.

## Conclusion

This project demonstrates the use of Bayesian linear regression models with different basis functions for
temperature prediction. By comparing the performance of various methods, we can determine the most effective approach 
for temperature prediction tasks.

## Course Information

This exercise is part of the course "Bayesian Machine Learning 67564" at the Hebrew University of Jerusalem.
The implementation and results are based on the course materials and guidelines provided by the instructors.

