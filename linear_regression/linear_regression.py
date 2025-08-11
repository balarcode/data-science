################################################
# Title     : Linear Regression
# Author    : balarcode
# Version   : 1.1
# Date      : 10th August 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Linear regression for multiple input features in a dataset is implemented in this source file.
#             Multiple input features means that the dataset has multiple dimensions.
#             The dataset is a labeled dataset which is considered as a training dataset in order to build a linear regression model.
#             Given a dataset (size = N x D), linear regression model is built using the following equation.
#               f_wb(x) = w . x + b, where 'w' is the weight vector and 'b' is the bias.
#             N is the number of data points in the dataset, 'x' and D is the dimension of the data point.
#
# All Rights Reserved.
################################################
# %%
import numpy as np
import matplotlib.pyplot as plt
import copy

# %%
# Load the dataset
data = np.loadtxt("./data/dataset_house_prices.txt", delimiter=',', skiprows=1)
X = data[:,:4]
y = data[:,4]
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# %%
# Plot the dataset across all the input features
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:, i], y, label='target')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("House Price (1000's)")
fig.suptitle("Plot of Dataset (House Price v/s Multiple Input Features) without Normalization")
plt.legend(loc="upper center")
plt.show()

# %%
################################################
# Function Definitions
################################################
def compute_cost(X, y, w, b):
    """Compute the cost as mean square error (MSE).
       Goal will be to minimize MSE w.r.t. weight and bias parameters.
    """
    m = X.shape[0] # Number of training examples or data points
    cost = 0.0 # Initial value for the cost
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i])**2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    """
    Compute the gradient for linear regression of a multi-dimensional dataset.
    dj_dw (1D vector) is the gradient of the cost w.r.t. the weight vector, w.
    dj_db (scalar) is the gradient of the cost w.r.t. the bias, b.
    Both dj_dw and dj_db are partial derivatives.
    """
    m, n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0

    error = (X @ w + b) - y
    dj_dw = (X.T @ error) / m
    dj_db = (np.sum(error)) / m

    return dj_db, dj_dw

def gradient_descent_algorithm(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iterations):
    """
    Execute batch gradient descent algorithm to learn weight vector, w and bias, b.
    Iteratively update w and b using gradient steps with learning rate, alpha.
    """
    J = [] # Cost
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iterations):
        # Compute the gradient
        # NOTE: At any given point, the computed gradient is a vector that points in the direction of the 
        # function's steepest local increase. It can be said that the gradient points to the maximum (i.e.
        # one of local maximas or global maximum).
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update parameters w and b using alpha and computed gradients
        # NOTE: Including a negative sign for the gradient directs the gradient vector towards the minimum
        # so that the function's local minima or global minimum is found when the gradient becomes zero.
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J for each iteration
        if i < 100000:
            J.append(cost_function(X, y, w, b))

    return w, b, J

def data_normalize(X):
    """
    Compute standardization or Z-score normalization.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

# %%
################################################
# Build Linear Regression Model for Multiple Features
################################################
# Input data normalization
X_norm, X_mu, X_sigma = data_normalize(X)

# Initialize weight vector, w and bias, b
m, n = X.shape
w_init = np.zeros((n, ))
b_init = 0.0

iterations = 10000 # Number of iterations
alpha = 1.0e-1 # Learning rate

# Run gradient descent algorithm
w_opt, b_opt, J_hist = gradient_descent_algorithm(X_norm, y, w_init, b_init,
                                                  compute_cost, compute_gradient,
                                                  alpha, iterations)

print(f"Gradient descent algorithm output - b: {b_opt}, w: {w_opt}")
# Loop over every training example or data point to confirm the linear regression model prediction
for i in range(m):
    print(f"Prediction: {np.dot(X_norm[i], w_opt) + b_opt}, Target Value: {y[i]}")

# %%
# Plot the linear regression model predictions and targets across all the input features
y_pred = (X_norm @ w_opt + b_opt)
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_norm[:,i], y, label='target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_norm[:,i], y_pred, color='orange', label='predict')
ax[0].set_ylabel("House Price (1000's)")
fig.suptitle("Target v/s Prediction using Normalized Linear Regression Model")
plt.legend(loc="best")
plt.show()

# Plot computed cost versus number of iterations
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(J_hist, color='green')
ax.set_title("Cost v/s Iteration for Normalized Linear Regression Model")
ax.set_ylabel('Cost')
ax.set_xlabel('Number of Iterations')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(np.arange(len(J_hist[0:100])), J_hist[0:100], color='green')
ax2.plot((iterations-100) + np.arange(len(J_hist[(iterations-100):])), J_hist[(iterations-100):], color='green')
ax1.set_title("Cost v/s Iteration (begin)");  ax2.set_title("Cost v/s Iteration (end)")
ax1.set_ylabel('Cost')                     ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('Number of Iterations')     ;  ax2.set_xlabel('Number of Iterations')
plt.show()
# %%
