################################################
# Title     : Sentiment Classifier
# Author    : balarcode
# Version   : 1.1
# Date      : 6th November 2024
# File Type : Python Script / Program
# File Test : Verified on Python 3.12.6
# Comments  : The twitter data in the csv file is a not a real data.
#             It is a fake semi-randomly generated data for usage in this program only.
#             The words that express positive and negative sentiments are provided
#             in files: positive_words.txt and negative_words.txt respectively.
#
# All Rights Reserved.
################################################

import os

# NOTE: Use the below code snippet to figure out the absolute path to the working directory.
#       Failure to do so will lead to program failure while reading files.
cwd = os.getcwd() # Get the current working directory
files = os.listdir(cwd) # Get all the files and sub-directories in that directory
print("Files in {}: {}".format(cwd, files))
working_directory = cwd + "/Python/sentiment_classifier/"
print("Working directory is: {}".format(working_directory))

punctuation_chars = ["'", '"', ",", ".", "!", ":", ";", "#", "@"]

################################################
# Function Definitions
################################################
def strip_punctuation(str):
    """Function to remove punctuation characters and return only alphabet words."""
    new_str = str
    for ch in str:
        if ch in punctuation_chars:
            new_str = str.replace(ch, '')
    return new_str

def get_pos(str):
    """Function to return the number of positive word occurrences."""
    words = str.lower().split()
    positive_integer = 0
    for word in words:
        word_wo_punctuation = strip_punctuation(word)
        if word_wo_punctuation in positive_words:
            positive_integer += 1
    return positive_integer

def get_neg(str):
    """Function to return the number of negative word occurrences."""
    words = str.lower().split()
    negative_integer = 0
    for word in words:
        word_wo_punctuation = strip_punctuation(word)
        if word_wo_punctuation in negative_words:
            negative_integer += 1
    return negative_integer

################################################
# Sentiment Classifier Logic
################################################
# Read and process the words that express positive sentiments
positive_words = []
with open(working_directory + "positive_words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            positive_words.append(lin.strip())

# Read and process the words that express negative sentiments
negative_words = []
with open(working_directory + "negative_words.txt") as pos_f:
    for lin in pos_f:
        if lin[0] != ';' and lin[0] != '\n':
            negative_words.append(lin.strip())

# Read and process input twitter data from file: project_twitter_data.csv
infile = open(working_directory + "project_twitter_data.csv", "r")
twitter_data = [] # List of tuples
net_score = 0
lines = infile.readlines()
print(lines[0]) # Print header
for line in lines[1:]: # Start after the header line
    print(line)
    vals = line.strip().split(',')
    positive_score = get_pos(vals[0])
    negative_score = get_neg(vals[0])
    net_score = (positive_score - negative_score)
    twitter_data.append((vals[1], vals[2], positive_score, negative_score, net_score))

# Populate results in file: sentiment_classifier_results.csv
outfile = open(working_directory + "sentiment_classifier_results.csv", "w")
outfile.write('Number of Retweets, Number of Replies, Positive Score, Negative Score, Net Score')
outfile.write('\n')
for data in twitter_data:
    row_string = '{}, {}, {}, {}, {}'.format(data[0], data[1], data[2], data[3], data[4])
    outfile.write(row_string)
    outfile.write('\n')
outfile.close()
