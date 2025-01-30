################################################
# Title     : Regularized Logistic Regression Algorithm
# Author    : balarcode
# Version   : 1.1
# Date      : 29th January 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Logistic regression algorithm is a classification algorithm. It tries to fit an
#             S-shaped curve to the datset and upon thresholding would classify the data points
#             as either a 0 or 1. The S-shaped curve is defined by the sigmoid function with
#             decision boundary given by a polynomial function.
#             The cost function used to minimize with respect to model parameters w and b is a
#             binary cross entropy function which is convex. The estimated parameters w and b
#             are maximum likelihood estimates that minimizes the cost function. Gradient descent
#             algorithm using computed gradients and learning rate is used to compute the parameters
#             w and b in the implementation.
#             (1) Regularization is performed to avoid overfitting of the data points from multiple
#             features. This helps to reduce the variance of the logistic regression algorithm.
#             (2) Increasing the number of features to include additional features such as higher order
#             polynomial terms will help the logistic regression algorithm to fit the dataset pretty
#             well with less bias. As a further improvement, selecting only the key features for the
#             given dataset will make the algorithm just right.
#
# All Rights Reserved.
################################################
# %%
# Import packages and libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

# %%
# Load the dataset
data = np.loadtxt("data/data.txt", delimiter=',')
X_training = data[:, :2]
y_training = data[:, 2]

print("X_training:", X_training[:5])
print("Type of X_training:",type(X_training))
N, D = X_training.shape
print('Number of training examples (N):', N)
print('Number of features (D):', D)

print("y_training:", y_training[:5])
print("Type of y_training:",type(y_training))

# %%
# Plot the labeled training examples
positive = y_training[:] == 1
negative = y_training[:] == 0

plt.plot(X_training[positive, 0], X_training[positive, 1], 'k+', label="Accepted")
plt.plot(X_training[negative, 0], X_training[negative, 1], 'yo', label="Rejected")
plt.ylabel('x2')
plt.xlabel('x1')
plt.title('Plot of Dataset with Baseline Features x1 and x2')
plt.legend(loc="upper right")
plt.show()

# %%
# Function Definitions
def map_features(X1, X2):
    """
    Feature mapping function to create additional features for each data point.
    In this function, each data point is mapped to polynomial features up to sixth order.
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

def sigmoid(z):
    """
    Computes the sigmoid of z.
    Also known as logistic function, the function returns a value in the range 0 to 1.
    """
    g = 1 / (1 + np.exp(-z))

    return g

def compute_cost(X, y, w, b):
    """
    Computes the cost for logistic regression algorithm over all training examples.
    Cost is computed as binary cross entropy or log loss.
    The function does not include the regularization term.
    """
    m, n = X.shape

    loss = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f_wb = sigmoid(z)
        loss += (-y[i] * np.log(f_wb)) - ((1 - y[i]) * np.log(1 - f_wb))
    total_cost = loss / m

    return total_cost

def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost for regularized logistic regression algorithm over all training examples.
    """
    m, n = X.shape

    cost_without_regularization = compute_cost(X, y, w, b)

    regularization_term = 0.

    for j in range(n):
        reg_cost_j = w[j]**2
        regularization_term += reg_cost_j
    regularization_term = (lambda_ / (2 * m)) * regularization_term

    # Regularization term is added to get the total cost
    total_cost = cost_without_regularization + regularization_term

    return total_cost

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for logistic regression algorithm.
    dj_dw (1D vector) is the gradient of the cost w.r.t. the weight vector, w.
    dj_db (scalar) is the gradient of the cost w.r.t. the bias, b.
    Both dj_dw and dj_db are partial derivatives.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    error = sigmoid(X @ w + b) - y
    dj_dw = (X.T @ error) / m
    dj_db = (np.sum(error)) / m

    return dj_db, dj_dw

def compute_gradient_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the gradient for regularized logistic regression algorithm.
    Regularization term (scalar) is added to every element in dj_dw (1D vector).
    """
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    # Regularization term is added for every feature in weight vector, w
    for j in range(n):
        dj_dw[j] = dj_dw[j] + ((lambda_ / m) * w[j])

    return dj_db, dj_dw

def gradient_descent_algorithm(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iterations, lambda_):
    """
    Execute batch gradient descent algorithm to learn or estimate weight vector,
    w and bias, b with regularization being applied.
    Iteratively update w and b using gradient steps and learning rate, alpha.
    """

    # Number of training examples
    m = len(X)

    w = copy.deepcopy(w_in)
    b = b_in
    J = [] # Cost
    w_history = [] # History of w updates

    for i in range(num_iterations):
        # Compute the gradient
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)

        # Update parameters w and b using alpha and computed gradients
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J for each iteration
        if i < 100000:
            J.append(cost_function(X, y, w, b, lambda_))

        # Print cost at every 1/10th of iterations or for reaching num_iterations
        if (i % math.ceil(num_iterations/10) == 0) or (i == (num_iterations-1)):
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J[-1]):8.2f}   ")

    return w, b, J, w_history

def predict(X, w, b):
    """
    Predicts whether the label is 0 or 1 using learned logistic
    regression parameters: weights, w and bias, b.
    The predictions for X use a threshold of 0.5 after applying
    the sigmoid logistic function.
    """
    m, n = X.shape
    p = np.zeros(m)

    # Loop over each training example
    for i in range(m):
        z_wb = 0
        # Loop over each feature
        for j in range(n):
            z_wb += X[i, j] * w[j]

        # Add the bias term
        z_wb += b

        # Calculate the prediction
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = (f_wb >= 0.5)

    return p

def plot_decision_boundary(w, b, X):
    """
    Plots the decision boundary for the transition from y = 0 to y = 1 predictions
    by the logistic regression model as a contour plot.
    Credit: dibgerge on Github.
    """
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        plt.plot(plot_x, plot_y, c="b")
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = sigmoid(np.dot(map_features(u[i], v[j]), w) + b)

        # Transpose of z
        z = z.T

        # Plot z
        plt.contour(u, v, z, levels=[0.5], colors="g")

# %%
# Regularized Logistic Regression Algorithm For Mapped Features (Higher Dimensions)

# Feature mapping
X_mapped = map_features(X_training[:, 0], X_training[:, 1])
print("Shape after feature mapping:", X_mapped.shape)

# Initialize algorithm parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_
lambda_ = 0.01

iterations = 10000 # Number of iterations
alpha = 0.01 # Learning rate

# Run gradient descent algorithm
w_opt, b_opt, J_hist, w_hist = gradient_descent_algorithm(X_mapped, y_training, initial_w, initial_b,
                                                          compute_cost_reg, compute_gradient_reg,
                                                          alpha, iterations, lambda_)

# %%
# Plot the regularized logistic regression model predictions along with decision boundary
# The decision boundary is where the predictions change from y = 0 to y = 1
positive = y_training[:] == 1
negative = y_training[:] == 0
plt.plot(X_mapped[positive, 0], X_mapped[positive, 1], 'k+', label="y=1")
plt.plot(X_mapped[negative, 0], X_mapped[negative, 1], 'yo', label="y=0")
plot_decision_boundary(w_opt, b_opt, X_mapped)
plt.ylabel('x2')
plt.xlabel('x1')
plt.title('Predictions from Regularized Logistic Regression Model')
plt.legend(loc="upper right")
plt.show()

# %%
# Plot computed cost versus number of iterations
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(J_hist, color='green')
ax.set_title("Cost v/s Iteration for Regularized Logistic Regression Model")
ax.set_ylabel('Cost')
ax.set_xlabel('Number of Iterations')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(np.arange(len(J_hist[0:2000])), J_hist[0:2000], color='green')
ax2.plot((iterations-2000) + np.arange(len(J_hist[(iterations-2000):])), J_hist[(iterations-2000):], color='green')
ax1.set_title("Cost v/s Iteration (begin)");  ax2.set_title("Cost v/s Iteration (end)")
ax1.set_ylabel('Cost')                     ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('Number of Iterations')     ;  ax2.set_xlabel('Number of Iterations')
plt.show()

# %%
# Compute the accuracy of the regularized logistic regression algorithm on the mapped training set
p = predict(X_mapped, w_opt, b_opt)

print('Training Accuracy: %f'%(np.mean(p == y_training) * 100))
# %%
