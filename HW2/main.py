import csv
import math
import random


# Import the training and test files into lists
def import_data(train, test):
    # Import the training file
    with open(train, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        global training_list
        training_list = list(reader)

    # Import the test file
    with open(test, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        global testing_list
        testing_list = list(reader)

    # Convert all values to ints
    for movie in range(len(training_list)):
        for index in range(len(training_list[movie])):
            training_list[movie][index] = int(training_list[movie][index])
        # Remove the timestamp
        del training_list[movie][-1]

    # Convert all values to ints
    for movie in range(len(testing_list)):
        for index in range(len(testing_list[movie])):
            testing_list[movie][index] = int(testing_list[movie][index])
        # Remove the timestamp
        del testing_list[movie][-1]


# Given list of movie ratings, group ratings based on which user rated them
def convert_to_user_dict(train, test):
    for rating in train:
        global user_training_dict
        if not (0 <= rating[0] - 1 & rating[0] - 1 < len(user_training_dict)):
            user_training_dict.update({rating[0]: {}})
        # rating[0] = userID
        # rating[1] = movieID
        # rating[2] = movieRating
        user_training_dict[rating[0]].update({rating[1]: rating[2]})

    for rating in test:
        global user_testing_dict
        if not (0 <= rating[0] - 1 & rating[0] - 1 < len(user_testing_dict)):
            user_testing_dict.update({rating[0]: {}})
        # rating[0] = userID
        # rating[1] = movieID
        # rating[2] = movieRating
        # dict.update() and = assignment do the same thing lol
        user_testing_dict[rating[0]][rating[1]] = rating[2]


# Calculate the cosine similarity between two users
# This ignores any movies that haven't been rated by both users
def cosine_similarity(user1, user2):
    # similarity = ( A dot B ) / ( ||A|| * ||B|| )
    # We only care about movies that both users have rated
    dot = 0.0
    magnitude1 = 0.0  # ||A||
    magnitude2 = 0.0  # ||B||
    for key1 in user1.keys():
        for key2 in user2.keys():
            # Only look at movies both users have rated
            if key1 == key2:
                dot += user1[key1] * user2[key2]
                magnitude1 += pow(user1[key1], 2)
                magnitude2 += pow(user2[key2], 2)
    magnitude1 = math.sqrt(magnitude1)
    magnitude2 = math.sqrt(magnitude2)

    if (magnitude1 * magnitude2) == 0.0:
        # Check for division by 0
        return 0.0
    else:
        similarity = dot / (magnitude1 * magnitude2)
        return similarity


# Return the k nearest neighbors to a given training user
# Also throws out training users that haven't rated a certain movie
def get_neighbors(user, train_dict, k, movieID):
    # Keys: userID
    # Values: similarity between given test user and userID
    nearest_neighbors = dict()
    for userID in train_dict.keys():
        # Only consider comparing users who have rated a movie we want to predict
        # And don't compare a user to itself
        if (movieID in train_dict[userID].keys()) & (user != userID):
            similarity = cosine_similarity(train_dict[user], train_dict[userID])
            nearest_neighbors[userID] = similarity
            # Only store k nearest neighbors
            if len(nearest_neighbors) < k:
                nearest_neighbors[userID] = similarity
            else:
                # NeighborID of the lowest similarity of the k neighbors currently stored
                min_similarity = 0
                for neighborID in nearest_neighbors.keys():
                    if min_similarity == 0:
                        min_similarity = neighborID
                    elif nearest_neighbors[neighborID] < nearest_neighbors[min_similarity]:
                        min_similarity = neighborID
                # If this similarity is greater than the smallest similarity of the k stored
                if similarity > nearest_neighbors[min_similarity]:
                    del nearest_neighbors[min_similarity]
                    nearest_neighbors[userID] = similarity
                    break
    return nearest_neighbors


# Using the nearest neighbors and their similarities as weights, predict the rating for a movie
def weighted_prediction(userID, train_dict, movieID):
    # Keys: Rating value (1 - 5 stars)
    # Values: Sum of similarities of users who gave the corresponding rating value
    predictions = dict()
    # Get k nearest neighbors
    nearest_neighbors = get_neighbors(userID, train_dict, k, movieID)
    for neighborID in nearest_neighbors.keys():
        if train_dict[neighborID][movieID] not in predictions.keys():
            # This prediction hasn't been initialized
            predictions[train_dict[neighborID][movieID]] = 0.0
        predictions[train_dict[neighborID][movieID]] = predictions[train_dict[neighborID][movieID]] + nearest_neighbors[neighborID]
    # max_rating is the 1-5 star rating and a key in the predictions dictionary
    # Set the default guess in the middle of the range [1, 5] to minimize error
    max_rating = 3
    for rating in predictions.keys():
        if max_rating not in predictions.keys():
            # max_rating isn't defined
            max_rating = rating
        elif predictions[rating] > predictions[max_rating]:
            max_rating = rating
    return max_rating


# Calculate error for one user
def calculate_error_for_user(userID, train_dict, test_dict):
    error = 0.0
    movie_count = 0
    for movieID in test_dict[userID].keys():
        movie_count += 1
        prediction = weighted_prediction(userID, train_dict, movieID)
        expected = test_dict[userID][movieID]
        error += expected - prediction
    # print "User", userID, "error:", error
    return error / movie_count


# Calculate mean squared error for all users
def mean_squared_error(train_dict, test_dict):
    count_users = 0
    sum_errors = 0.0
    for userID in test_dict.keys():
        count_users += 1
        sum_errors += pow(calculate_error_for_user(userID, train_dict, test_dict), 2)
    return sum_errors / count_users


# Convert a list of lists into a dictionary of dictionaries
def convert_list_to_dict(fold_list):
    dictionary = dict(dict())
    for fold in fold_list:
        for rating in fold:
            if rating[0] not in dictionary.keys():
                dictionary.update({rating[0]: {}})
            # rating[0] = userID
            # rating[1] = movieID
            # rating[2] = movieRating
            dictionary[rating[0]][rating[1]] = rating[2]
    return dictionary


# Given a list of folds and an index, combine dictionaries into training and test sets
def combine_folds(folds, index):
    list_of_folds = []
    training_set = dict()
    validation_set = dict()
    # Make index-th fold validation_set and all others combine into training_set
    for fold in range(len(folds)):
        if fold == index:
            validation_set = convert_list_to_dict([folds[fold]])
        else:
            list_of_folds.append(folds[fold])
    training_set = convert_list_to_dict(list_of_folds)
    return training_set, validation_set


# Run k-fold cross validation over a user dictionary split into k number of folds
def k_fold_cross_validation(num_folds):
    list_of_folds = list()
    # Get all ratings and shuffle them
    ratings = training_list
    random.shuffle(ratings)
    size = len(ratings) / num_folds
    # Build all folds
    for fold in range(num_folds):
        current_fold = list()
        count = 0
        # Loop over all triples in the keys list
        for triple in ratings:
            # Stop when current fold is full
            if count == size:
                break
            current_fold.append(triple)
            count += 1
        list_of_folds.append(current_fold)

    avg = 0.0
    for fold in range(num_folds):
        training_set, validation_set = combine_folds(list_of_folds, fold)
        error = mean_squared_error(training_set, validation_set)
        avg += error
        print "\tMean square error for validation set", fold + 1, ":", error
        # print "\tFold", fold + 1, "done"
    print "\t\tAverage mean square error for folds =", num_folds, ":", avg / num_folds


def main():
    import_data(training_file, testing_file)
    convert_to_user_dict(training_list, testing_list)

    # For when k = 3, find mean squared error on given test data
    print "Mean squared error for number of neighbors k = 3:", mean_squared_error(user_training_dict, user_testing_dict)

    # Cross-validation using k-folds
    # Iterate between [2, 4, 6, 8, 10] folds
    for num_folds in range(2, 11, 2):
        print "\nNumber of folds =", num_folds
        # Iterate over [1, 3, 5, 7, 9] neighbors
        for num_neighbors in range(1, 10, 2):
            global k
            k = num_neighbors
            print "Number of neighbors k =", k
            k_fold_cross_validation(num_folds)


training_file = 'data/u1-base.base'
testing_file = 'data/u1-test.test'
k = 3
training_list = list()
testing_list = list()
user_training_dict = dict(dict())
user_testing_dict = dict(dict())

main()
