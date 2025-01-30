## Collaborative Filtering Algorithm for a Recommender System

### Updated Recommendations for a New User

After the new user has watched few movies including some of the recommended ones from the initial recommendation list and has rated those movies, the recommender system then updates the recommendation list for the new user by rerunning the collaborative filtering algorithm incorporating the new user's new movie ratings. It can be seen that the recommender system has improved in terms of the predicted movie ratings and is able to recommend more movies of similar ratings to the new user. This was possible by learning W and b parameters specific to the new user.

[Updated Recommendations](results/updated_recommendations_for_new_user.csv)

### Initial Recommendations for a New User

The new user is yet to watch and rate movies in the movie catalogue. The recommender system provides a list of initial recommendations that the new user might want to watch based on the learning from previously provided movie ratings by different users to the movies in the movie catalogue.

[Initial Recommendations](results/initial_recommendations_for_new_user.csv)

### Training Loss of the Model for Updated Recommendations

- Training loss at iteration 0: 2321306.4
- Training loss at iteration 20: 136156.8
- Training loss at iteration 40: 51863.4
- Training loss at iteration 60: 24601.0
- Training loss at iteration 80: 13632.7
- Training loss at iteration 100: 8489.4
- Training loss at iteration 120: 5808.9
- Training loss at iteration 140: 4312.6
- Training loss at iteration 160: 3436.2
- Training loss at iteration 180: 2903.0

### Training Loss of the Model for Initial Recommendations

- Training loss at iteration 0: 2287998.9
- Training loss at iteration 20: 132726.4
- Training loss at iteration 40: 49977.3
- Training loss at iteration 60: 23645.4
- Training loss at iteration 80: 13123.3
- Training loss at iteration 100: 8205.2
- Training loss at iteration 120: 5642.8
- Training loss at iteration 140: 4209.9
- Training loss at iteration 160: 3368.3
- Training loss at iteration 180: 2854.8


## Citation

Please note that the code and technical details shared are for educational purposes only. The repo is not open for collaboration.

If you happen to use the code from this repo, please cite my user name along with link to my profile: https://github.com/balarcode. Thank you!
