################################################
# Title     : Collaborative Filtering Algorithm
# Author    : balarcode
# Version   : 1.0
# Date      : 30th January 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Collaborative filtering algorithm is used for Recommender Systems. The algorithm
#             learns about the ratings provided for items by users to predict a rating for a
#             new user who uses the same set of items from the catalogue. The items can include
#             a catalogue of movies, apparels, food, etc.
#             Multiple users contribute via collaboration in building a recommender system. By
#             learning from this built system, prediction or inference can be made on how new
#             users could rate the same items and thereby recommending the highly rated items
#             as a recommendation to the new users.
#             Prediction of a rating for a new user, j for a new item, i is given by:
#               w(j) . x(i) + b(j)
#             where, w(j) and b(j) are parameters learnt by fitting a regularized regression
#             model for all valid ratings provided by n_u number of users.
#             where, x(i) is the feature vector for every item, i learnt by fitting a regularized
#             regression model for all valid ratings received for n_i number of items.
#             In summary, parameters w(j), b(j) and x(i) for j = 1 to n_u and i = 1 to n_i that
#             minimize the cost function is learnt by the collaborative filtering algorithm before
#             making newer predictions or inferences.
#             Dataset was obtained from https://grouplens.org/datasets/movielens/latest/.
#
# All Rights Reserved.
################################################
# %%
# Import packages and libraries
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# %%
# Load the dataset
# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files locally.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
working_directory = cwd + "/"
# print(working_directory)

# Matrix of feature vectors x(i) for each movie, i
# i-th row of X corresponds to the feature vector x(i) for the i-th movie (dimension = n)
file1 = open(working_directory + 'data/movies_X.csv', 'rb')
X = np.loadtxt(file1, delimiter=",")

# Number of movies, n_m and number of features, n
# NOTE: Movies are items in this dataset. The subscript 'm' is used instead of 'i' to specifically
# indicate number of movies.
n_m, n = X.shape

# Matrix of parameter vectors, w(j) for each user, j
# j-th row of W corresponds to one parameter vector w(j) for the j-th user (dimension = n)
file2 = open(working_directory + 'data/movies_W.csv', 'rb')
W = np.loadtxt(file2, delimiter=",")

# Number of users, n_u
n_u, _ = W.shape

# Vector of parameter scalars, b(j) for each user, j
file3 = open(working_directory + 'data/movies_b.csv', 'rb')
b = np.loadtxt(file3, delimiter=",")
b = b.reshape(1, -1)

# Ratings matrix, Y(i, j) of size = n_m X n_u
file4 = open(working_directory + 'data/movies_Y.csv', 'rb')
Y = np.loadtxt(file4, delimiter=",")

# Binary valued indicator matrix, R(i, j)
# where a value of 1 indicates if the user j gave a valid rating to movie i
# where a value of 0 indicates no valid rating was given to movie i
# Valid rating value to be retrieved from Y(i, j) corresponding to R(i, j) = 1
file5 = open(working_directory + 'data/movies_R.csv', 'rb')
R = np.loadtxt(file5, delimiter=",")

# Movies data frame and index of movies in the order they are in the Y ratings matrix
movie_list_df = pd.read_csv(working_directory + 'data/movie_list.csv', header=0, index_col=0, delimiter=',', quotechar='"')
movie_list = movie_list_df["title"].to_list()

print("Number of features: "    , n)
print("Number of movies rated: ", n_m)
print("Number of users: "       , n_u)

# %%
# Function definitions
def compute_cost(X, W, b, Y, R, lambda_):
    """
    Computes the cost for collaborative filtering algorithm.
    Regularization term is added to the mean square error term.
    """
    error = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
    J = 0.5 * tf.reduce_sum(error**2) + (lambda_ / 2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J

def gradient_descent_algorithm(X, W, b, Ynorm, R, num_iterations, lambda_):
    """
    Execute batch gradient descent algorithm to learn or estimate W, b and X
    parameters with regularization being applied.
    Repeat until convergence:
    - Compute forward pass
    - Compute gradients i.e. partial derivatives of the cost w.r.t. the parameters
    - Update the parameters using the learning rate and the computed gradients
    """
    for iter in range(num_iterations):
        # Use TensorFlow’s GradientTape to record the operations used to compute the cost
        with tf.GradientTape() as tape:

            # Compute the cost (forward pass included in cost)
            cost_value = compute_cost(X, W, b, Ynorm, R, lambda_)

        # Using the gradient tape, automatically retrieve the gradients of the
        # parameters trained with respect to the cost function (average loss)
        gradients = tape.gradient(cost_value, [X, W, b])

        # Run one step of gradient descent using alpha and computed gradients
        # to update the parameters to minimize the cost function (average loss)
        optimizer.apply_gradients(zip(gradients, [X, W, b]))

        # Print cost for every 20 iterations
        if iter % 20 == 0:
            print(f"Training loss at iteration {iter}: {cost_value:0.1f}")

    return X, W, b

# %%
# Collaborative Filtering Algorithm

# Normalize the dataset
Ymean = (np.sum(Y*R, axis=1) / (np.sum(R, axis=1)+1e-12)).reshape(-1, 1)
Ynorm = Y - np.multiply(Ymean, R)

iterations = 200 # Number of iterations
lambda_ = 1 # Regularization parameter
alpha = 1e-1 # Learning rate

# Initialize the parameters for gradient descent algorithm as Tensor variables with random values
tf.random.set_seed(1234)
num_movies, num_users = Y.shape
num_features = 100 # Train for a large number of features more than the baseline features in 'n'
W = tf.Variable(tf.random.normal((num_users,  num_features), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1,          num_users),    dtype=tf.float64), name='b')

# Instantiate the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)

# Run gradient descent algorithm
X, W, b = gradient_descent_algorithm(X, W, b, Ynorm, R, iterations, lambda_)

# %%
# Prediction or Inference

# Predict ratings using learned parameters: weights W, biases b and feature vectors in X
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

# Restore the mean which was normalized earlier
pm = p + Ymean

predictions = pm[:, 0]

# Sort predictions
ix = tf.argsort(predictions, direction='DESCENDING')

filter = (movie_list_df["number of ratings"] > 20)
movie_list_df["predictions"] = predictions
movie_list_df = movie_list_df.reindex(columns=["predictions", "mean rating", "number of ratings", "title"])
movie_list_df.loc[ix[:]].loc[filter].sort_values("mean rating", ascending=False)
# %%
