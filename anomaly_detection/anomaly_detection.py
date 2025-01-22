################################################
# Title     : Anomaly Detection Algorithm
# Author    : balarcode
# Version   : 1.0
# Date      : 22nd January 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Anomaly detection algorithm is an unsupervised machine learning algorithm which learns from
#             the dataset of normal events to raise a red flag when it detects unusual or anomalous event(s).
#             The training dataset with multiple features (i.e. higher dimensions) is fit to a multivariate
#             Gaussian distribution or joint normal distribution using parameters mean and variance estimated
#             for each of the features in the training dataset. Then, the threshold for anomaly detection also
#             known as epsilon is deduced using cross validation dataset to complete the anomaly detection
#             algorithm for potential use with real world data.
#
# All Rights Reserved.
################################################
# %%
# Import packages and libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# %%
# Load the dataset
X_training = np.load("data/X_training.npy") # Unlabeled dataset
X_validation = np.load("data/X_validation.npy") # Labeled dataset
y_validation = np.load("data/y_validation.npy") # Ground truth labels (small in number)

print ('The shape of X in training set is:', X_training.shape)
m, n = X_training.shape
print('Number of training examples (m):', m)
print('Number of features (n):', n)
print('The shape of X in cross validation set is:', X_validation.shape)
print('The shape of y (ground truth labels) in cross validation set is:', y_validation.shape)

# %%
# Function Definitions
def compute_parameters(X):
    """
    Computes mean and variance of all features in the dataset.
    """

    m, n = X.shape
    mu = np.mean(X, axis=0)
    var = np.sum(((X - mu)**2), axis=0) / m

    return mu, var

def multivariate_gaussian(X, mu, var):
    """
    Computes the probability density function of the dataset with
    multiple features as a multivariate Gaussian distribution or
    joint normal distribution with parameters, mu and var estimated
    for all features.
    """

    k = len(mu)

    # Consider var as a covariance matrix
    if var.ndim == 1:
        var = np.diag(var)

    X = (X - mu)
    pdf = (2 * np.pi)**(-k/2) * \
          np.linalg.det(var)**(-0.5) * \
          np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

    return pdf

def compute_epsilon(y, p_val):
    """
    Computes epsilon (threshold) to be used for anomaly detection to detect outliers or anomalies.
    It uses probability values from cross validation set and ground truth labels from cross validation set.
    """

    best_epsilon = 0 # Threshold for anomaly detection
    best_F1 = 0
    F1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
        predictions = (p_val < epsilon)
        true_positives = np.sum((predictions == 1) & (y == 1))
        false_positives = np.sum((predictions == 1) & (y == 0))
        false_negatives = np.sum((predictions == 0) & (y == 1))
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

def norm_plot(ax, data):
    scale = (np.max(data) - np.min(data)) * 0.2
    x = np.linspace(np.min(data)-scale, np.max(data)+scale, 50)
    _,bins, _ = ax.hist(data, x, color="xkcd:azure")

    mu = np.mean(data)
    sigma = np.std(data)
    # print(f"mu: {mu}, var: {sigma**2}")
    dist = norm.pdf(bins, loc=mu, scale=sigma)

    axr = ax.twinx()
    axr.plot(bins, dist, color="orange", lw=2)
    axr.set_ylim(bottom=0)
    axr.axis('off')

# %%
# Plot the input features and their probability distributions across all training examples
count = 0
colors = {0: 'b', 1: 'r', 2: 'g', 3:'c', 4:'m', 5:'y'}
while (count < n):
    var = "x_"+str(count)
    pvar = "p(x_"+str(count)+")"
    # fig, ax = plt.subplots(figsize=(3, 3), sharey=True)
    # ax.plot(X_training[:, count], color=colors[count%6])
    # ax.set_title("Feature v/s Number of Training Examples")
    # ax.set_ylabel(var)
    # ax.set_xlabel('Number of Training Examples')
    # plt.show()
    fig, ax = plt.subplots(figsize=(3, 3), sharey=True)
    norm_plot(ax, X_training[:, count], )
    ax.set_ylabel(pvar)
    ax.set_xlabel(var)
    fig.suptitle("Univariate Gaussian Distribution of Feature - "+var)
    plt.show()
    count += 1

# %%
# Design Anomaly Detection Algorithm For Multiple Features (Higher Dimensions)

# Compute mean and variance of all features from the training set
mu, var = compute_parameters(X_training)

# Compute probability density function of training set with mean and variance of all features from the training set
pdf_training = multivariate_gaussian(X_training, mu, var)

# Compute probability density function of cross validation set with mean and variance of all features from the training set
pdf_validation = multivariate_gaussian(X_validation, mu, var)

# Compute epsilon (threshold) to be used for anomaly detection
epsilon, F1 = compute_epsilon(y_validation, pdf_validation)

print('Epsilon found using cross validation set: %e'% epsilon)
print('F1 score on cross validation set:  %f'% F1)
print('Number of Anomalies found: %d'% sum(pdf_training < epsilon))
# %%
