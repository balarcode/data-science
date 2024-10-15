################################################
# Title     : K-Means Clustering
# Author    : balarcode
# Version   : 1.0
# Date      : 15th October 2024
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : The input data set is provided in plants.csv file and it contains
#             measurements of several species of plants.
#             Using data visualization feedback, the K-means clustering algorithm
#             is designed and implemented specifically for the provided data set.
#             The visualized charts are saved in the respective HTML files and the
#             results from K-means clustering is saved in a csv file.
#             The implemented code can be executed as one single file or cell-based
#             in conjunction with a Python interactive window.
#             Installation of Altair library: pip install altair
#             Installation of Vega datasets package: pip install vega_datasets
#
# All Rights Reserved.
################################################
# %%
import os
import math
import altair as alt
import pandas as pd

DEBUG = 0 # Debug flag to control printing of useful logs to the console

# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
print("Files in {}: {}".format(cwd, files))
#working_directory = cwd + "/Python/k_means_clustering/"
working_directory = cwd + "/" # Uncomment this line if running the code cell-based sequentially
print("Working directory is: {}".format(working_directory))

# %%
################################################
# Function Definitions
################################################
def calculate_distance(data_point_X, data_point_Y):
    """Compute cluster variance or squared Euclidean distance and return the square root of it."""
    squared_euclidean_distance = ((data_point_X[0] - data_point_Y[0]) ** 2) + ((data_point_X[1] - data_point_Y[1]) ** 2)
    return math.sqrt(squared_euclidean_distance)

def calculate_centroid(data_points):
    """Compute and return the mean or centroid of a cluster."""
    accummulator_X = 0
    accummulator_Y = 0
    number_of_data_points = len(data_points)
    for data_point in data_points:
        accummulator_X += data_point[0]
        accummulator_Y += data_point[1]
    mean_X = accummulator_X / number_of_data_points
    mean_Y = accummulator_Y / number_of_data_points
    return [mean_X, mean_Y]

# %%
################################################
# Data Visualization
################################################
infile = open(working_directory + "plants.csv", "r")
lines = infile.readlines()

# Form a Pandas data frame
plants_data = {}
params = lines[0].strip().split(',')
for idx in range(len(params)):
    plants_data[params[idx]] = []
for line in lines[1:]:
    vals = line.strip().split(',')
    for idx in range(len(vals)):
        plants_data[params[idx]].append(vals[idx])
plants_data_pd = pd.DataFrame(plants_data)

# Data visualization for input plants data set
plants_chart_width = alt.Chart(plants_data_pd).mark_point().encode(
    x='SepalWidthCm',
    y='PetalWidthCm',
    shape='Id',
    color='Id:N'
).properties(
    title='Plant Flower Widths'
)

plants_chart_width
plants_chart_width.save('plants_chart_width.html') # Write the chart into a HTML file

# %%
################################################
# K-means Clustering Algorithm
################################################
N = 3 # Number of clusters chosen from data visualization
centroids = [[3.4, 0.2], [2.7, 1.3], [3.0, 2.1]] # Initial centroid for each cluster based on data visualization
number_of_x_coordinates = len(plants_data['SepalWidthCm'])
number_of_y_coordinates = len(plants_data['PetalWidthCm'])
assert(number_of_x_coordinates == number_of_y_coordinates)
print("Number of Clusters from Data Visualization: {}".format(N))
print("Initial Centroids from Data Visualization: {}".format(centroids))
if DEBUG : print("Number of Data Points: {}".format(number_of_x_coordinates))

# For the chosen centroids from data visualization step, apply k-means clustering algorithm to categorize the data set into
# different clusters and update the centroids until convergence is met
converged = False
iterations = 0
while not converged:
    clusters = [[] for _ in range(N)]
    k_means_clustered_data_set = {'ClusterId' : [], 'Centroid' : [], 'Id' : [], 'SepalWidthCm' : [], 'PetalWidthCm' : []}
    for coordinate in range(number_of_x_coordinates):
        distance = []
        for cluster_idx in range(N):
            data_point = [float(plants_data['SepalWidthCm'][coordinate]), float(plants_data['PetalWidthCm'][coordinate])]
            distance.append(calculate_distance(data_point, centroids[cluster_idx]))
            if DEBUG : print("{} : {}".format(centroids[cluster_idx], data_point))
        if DEBUG : print(distance)
        cluster_id = distance.index(min(distance))
        clusters[cluster_id].append(data_point)
        k_means_clustered_data_set['ClusterId'].append(str(cluster_id))
        k_means_clustered_data_set['Centroid'].append(centroids[cluster_id])
        k_means_clustered_data_set['Id'].append(plants_data['Id'][coordinate])
        k_means_clustered_data_set['SepalWidthCm'].append(plants_data['SepalWidthCm'][coordinate])
        k_means_clustered_data_set['PetalWidthCm'].append(plants_data['PetalWidthCm'][coordinate])
        if DEBUG : print("Minimum Distance: {}, Cluster ID: {}".format(min(distance), cluster_id))
    centroids_updated = [calculate_centroid(cluster) for cluster in clusters]
    if DEBUG : print("Updated Centroids: {}".format(centroids_updated))
    converged = (centroids_updated == centroids)
    centroids = centroids_updated
    iterations += 1
    if converged == True:
        print("Number of Iterations Executed Until Convergence: {}".format(iterations))
        break

# Data visualization for k-means clustered data set
k_means_clustered_data_pd = pd.DataFrame(k_means_clustered_data_set)
k_means_clustered_data_chart = alt.Chart(k_means_clustered_data_pd).mark_point().encode(
    x='SepalWidthCm',
    y='PetalWidthCm',
    shape='ClusterId',
    color='ClusterId:N'
).properties(
    title='Plant Flower Widths after K-Means Clustering'
)

k_means_clustered_data_chart
k_means_clustered_data_chart.save('k_means_clustered_data_chart.html')

# Populate the results in file: k_means_clustered_results.csv
outfile = open(working_directory + "k_means_clustered_results.csv", "w")
outfile.write('ClusterId, Centroid, Id, SepalWidthCm, PetalWidthCm')
outfile.write('\n')
for idx in range(len(k_means_clustered_data_set['ClusterId'])):
    row_string = '{}, {}, {}, {}, {}'.format(k_means_clustered_data_set['ClusterId'][idx], k_means_clustered_data_set['Centroid'][idx], k_means_clustered_data_set['Id'][idx], k_means_clustered_data_set['SepalWidthCm'][idx], k_means_clustered_data_set['PetalWidthCm'][idx])
    outfile.write(row_string)
    outfile.write('\n')
outfile.close()

# %%
