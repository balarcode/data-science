################################################
# Title     : Principal Component Analysis
# Author    : balarcode
# Version   : 1.0
# Date      : 24th November 2024
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : Given a dataset (size = N x D), principal component analysis algorithm is applied to reduce
#             the dimension of the dataset onto a principal subspace spanned by the principal component
#             vectors as the basis. The problem is further solved in the principal subspace.
#             Here, N is the number of data points in the dataset and D is the dimension of the data point.
#
# All Rights Reserved.
################################################
# %%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# %%
def normalize(X):
    """Normalize the given dataset X to have zero mean and unit variance."""
    N, D = X.shape
    mu = np.sum(X, axis=0) / N
    sigma = np.sqrt(np.sum((X - mu) ** 2, axis=0) / N)
    Xnorm = (X - mu)/sigma
    return Xnorm, mu, sigma

# %%
def eigen_decomposition(S):
    """Compute the eigenvalues and corresponding eigenvectors for the covariance matrix S."""
    eigvals, eigvecs = np.linalg.eig(S)

    # Sort the eigenvalues and eigenvectors in descending order according to the eigenvalues
    sort_indices = np.argsort(eigvals)[::-1] # Default is quicksort in NumPy v2.1 implementation

    return eigvals[sort_indices], eigvecs[:, sort_indices]

# %%
def projection_matrix(B):
    """Compute the projection matrix for the principal subspace spanned by the basis vectors in matrix, B."""
    return B @ np.linalg.inv(B.T @ B) @ B.T

# %%
def pca_algorithm(X, num_components):
    """Using the PCA algorithm, reconstruct the dataset in a lower dimensional principal subspace.
       Use the number of principal components as an input to reconstruct the dataset.
    """
    # Normalization of the dataset
    N, D = X.shape
    X_normalized, mean, sigma = normalize(X)

    # Compute the data covariance matrix, S
    S = (X_normalized.T @ X_normalized) / N

    # Find the eigenvalues and corresponding eigenvectors for S
    eig_vals, eig_vecs = eigen_decomposition(S)

    # Choose the principal values and principal components of principal subspace
    indices = range(num_components)
    principal_values, principal_components = eig_vals[indices], eig_vecs[:, indices]

    # Due to precision error, the eigenvectors might come out to be complex, so only take their real parts
    principal_components = np.real(principal_components)

    # Reconstruct the datapoints after projecting the normalized dataset onto the basis spanned by the principal components
    X_reconstructed = projection_matrix(principal_components) @ X_normalized.T # Dot product chosen for inner product
    X_reconstructed = (X_reconstructed * np.array([sigma]).T) + np.array([mean]).T
    X_reconstructed = X_reconstructed.T
    return X_reconstructed, mean, sigma, principal_values, principal_components

# %%
def draw_vector(v0, v1, ax=None, label=None):
    """Draw a vector from v0 to v1."""
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0,
                    color='k')
    ax.annotate('', v1, v0, arrowprops=arrowprops, label=label)

# %%
# Principal component analysis for a 2D random (multivariate normal random variable) dataset
mvn = scipy.stats.multivariate_normal(
    mean=np.array([1, 1]),
    cov=np.array([[1, 0.8], [0.8, 1]])
)

# X contains 100 data points
X = mvn.rvs((100,), random_state=np.random.RandomState(0))

# Plot the original dataset
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], label='original data')
plt.axis('equal')
plt.legend()
ax.set(xlabel='$x_0$', ylabel='$x_1$')
plt.show()

# %%
# Plot the original dataset, reconstructed data points and the principal component vector
num_components = 1
X_reconstructed, mean, sigma, principal_values, principal_components = pca_algorithm(X, num_components)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(X[:, 0], X[:, 1], label='original data')
for (principal_variance, principal_component) in (zip(principal_values, principal_components.T)):
    draw_vector(mean, mean + np.sqrt(principal_variance) * principal_component, ax=ax)
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], label='reconstructed data')
plt.axis('equal')
plt.legend()
ax.set(xlabel='$x_0$', ylabel='$x_1$')
plt.show()
# %%
