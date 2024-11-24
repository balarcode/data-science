################################################
# Title     : K-Nearest Neighbors Classification
# Author    : balarcode
# Version   : 1.1
# Date      : 23rd November 2024
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Given a training set (matrix of real numbers of size = N x D) and
#             labels for the training set (vector of real numbers of size = N),
#             a label is predicted for a single test data point (size = D) as the
#             label of the majority of its "K-nearest neighbors" by using a distance measure.
#             Here, N is the number of data points in the dataset and D is the dimension of the data point.
#
# All Rights Reserved.
################################################
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from collections import Counter

# %%
# Load the Iris dataset
iris = datasets.load_iris()
training_data_set = iris.data[:, :2] # Training dataset
training_label_set = iris.target # Labels corresponding to the training dataset
# NOTE: First two dimensions (sepal length and sepal width) of Iris dataset is chosen as features to classify the flowers.

# %%
# Scatter plot of training dataset
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']) # List of colors as RGB values
cmap_bold = ListedColormap(['#FF0000',  '#00FF00', '#0000FF'])

fig, ax = plt.subplots(figsize=(4, 4))
for i, iris_class in enumerate(['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']):
    idx = (training_label_set==i)
    ax.scatter(training_data_set[idx, 0], training_data_set[idx, 1],
               c=cmap_bold.colors[i], edgecolor='k', s=20, label=iris_class)
ax.set(xlabel='sepal length (cm)', ylabel='sepal width (cm)')
ax.legend() # Per Iris class labels

# %%
def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between data points of X and Y"""
    X_minus_Y = X[:, np.newaxis] - Y # Find the pairwise difference between data points of X and Y
    distance_matrix = np.sqrt(np.sum(X_minus_Y**2, axis=2)) # Pairwise Euclidean distance
    return distance_matrix

# %%
def k_nearest_neighbors(K, training_data_set, training_label_set, test_data_point):
    """K-nearest neighbors algorithm implementation using pairwise distance matrix.
    'K' is the number of nearest neighbors to consider.
    test_data_point is a single test data point for which the label needs to be
    predicted using K-nearest neighbors."""

    # Compute the pairwise distance between the single test data point and all data points in training_data_set
    pairwise_distance = pairwise_distance_matrix(np.array([test_data_point]), training_data_set)

    # Find the indices of the K-nearest neighbors after sorting the distances in ascending order along second axis (across)
    K_nearest_indices = np.argsort(pairwise_distance, axis=1)[:, : K] # Default is quicksort in NumPy implementation

    # Find the labels of the K-nearest neighbors
    K_nearest_labels = training_label_set[K_nearest_indices].flatten()

    # Predict the most common or majority label for the test data point
    # NOTE: most_common(1) returns a list containing the predicted label and its count among the K_nearest_labels.
    prediction = Counter(K_nearest_labels).most_common(1)[0][0]

    return prediction

# %%
x_min, x_max = training_data_set[:, 0].min() - 0.3, training_data_set[:, 0].max() + 0.3
y_min, y_max = training_data_set[:, 1].min() - 0.3, training_data_set[:, 1].max() + 0.3
step = 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

prediction = [] # Predicted label for the test data point using K-nearest neighbors algorithm
K = 3
# Iteratively choose the test data point within the created mesh grid
for test_data_point in np.array([xx.ravel(), yy.ravel()]).T:
    prediction.append(k_nearest_neighbors(K, training_data_set, training_label_set, test_data_point))

fig, ax = plt.subplots(figsize=(4, 4))
# Overlay predicted labels as a colored mesh below the scattered training dataset
ax.pcolormesh(xx, yy, np.array(prediction).reshape(xx.shape), cmap=cmap_light) # Draw a colored plot within the created mesh grid
ax.scatter(training_data_set[:, 0], training_data_set[:, 1],
           c=training_label_set, cmap=cmap_bold, edgecolor='k', s=20)
ax.set(xlabel='sepal length (cm)', ylabel='sepal width (cm)')
# %%
