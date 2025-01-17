################################################
# Title     : Decision Tree Learning Algorithm
# Author    : balarcode
# Version   : 1.0
# Date      : 16th January 2025
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : A decision tree represents a function that takes a vector of feature values as an input and returns a "decision"
#             which would be a single output value. The input and output values can be discrete or continuous.
#             Building a decision tree requires learning from the structured dataset using algorithmic steps.
#             For inference, a decision tree would output the predicted or inferred value at the leaf node.
#             In this file, a decision tree learning algorithm is implemented for a multi-dimensional structured dataset considering
#             one hot encoding of the dataset and using concepts of entropy, information gain and optimal node splitting.
#             A discrete labeled dataset of three features for each training example is considered in order to build the decision
#             tree in this file. The three features are denoted as 0, 1 and 2.
#
# All Rights Reserved.
################################################

# Import packages and libraries
import numpy as np
import matplotlib.pyplot as plt

# One hot encoded dataset
# X_train contains three features for each of the training example
X_train = np.array([[1,1,1],
                    [1,0,1],
                    [1,0,0],
                    [1,0,0],
                    [1,1,1],
                    [0,1,1],
                    [0,0,0],
                    [1,0,1],
                    [0,1,0],
                    [1,0,0]])

# y_train contains the ground truth labels (either 1 or 0) for each of the training examples
y_train = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0])

print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))

################################################
# Function Definitions
################################################
def compute_entropy(y):
    """
    Computes the entropy for a node or split-branch in the decision tree.
    """
    entropy = 0.
    if (len(y) == 0):
        return 0
    else:
        p1 = len(y[y==1]) / len(y)
        if ((p1 == 0) or (p1 == 1)):
            return 0
        else:
            p0 = (1 - p1)
            entropy = - (p1 * np.log2(p1)) - (p0 * np.log2(p0))
    return entropy

def split_dataset(X, node_indices, feature):
    """
    Splits the data at a given node in the decision tree into left and right branches.
    """
    left_indices = []
    right_indices = []
    # Loop over indices or training examples active at this node for the input feature
    for i in node_indices:
        if (X[i][feature] == 1):
            left_indices.append(i) # Indices with feature value == 1
        else:
            right_indices.append(i) # Indices with feature value == 0
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    """
    Computes the information gain of splitting a node on a given feature in the decision tree.
    """
    # Split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Create intermediate variables to store features and labeled data for indexed training examples
    X_node, y_node = X[node_indices], y[node_indices] # Node
    X_left, y_left = X[left_indices], y[left_indices] # Left branch
    X_right, y_right = X[right_indices], y[right_indices] # Right branch

    information_gain = 0

    # Compute the information gain
    H_p1_node = compute_entropy(y_node) # Entropy of the node
    H_p1_left = compute_entropy(y_left) # Entropy of left branch upon node split
    w_left = len(X_left) / len(X_node)
    H_p1_right = compute_entropy(y_right) # Entropy of right branch upon node split
    w_right = len(X_right) / len(X_node)
    weighted_average_entropy = (w_left * H_p1_left) + (w_right * H_p1_right)
    information_gain = (H_p1_node - weighted_average_entropy)

    return information_gain

def get_best_split(X, y, node_indices):
    """
    Returns the optimal feature to split the node data in a decision tree.
    """
    num_features = X.shape[1]
    best_feature = -1 # Index of the best feature to split at a node
    max_info_gain = 0. # Zero is when there is no information gain
    for feature in range(num_features):
        info_gain_per_feature = compute_information_gain(X, y, node_indices, feature)
        if (info_gain_per_feature > max_info_gain):
            max_info_gain = info_gain_per_feature
            best_feature = feature
    return best_feature

def decision_tree(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a single decision tree using the recursive algorithm that splits
    the dataset into two branches at each node.
    """
    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return

    # Otherwise, get the best feature and split the data
    best_feature = get_best_split(X, y, node_indices)

    # Split the dataset w.r.t. the best feature
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))

    # Continue splitting the left and right child nodes
    decision_tree(X, y, left_indices, "Left", max_depth, current_depth+1)
    decision_tree(X, y, right_indices, "Right", max_depth, current_depth+1)

################################################
# Build the Decision Tree
################################################
tree = []
root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
decision_tree(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)