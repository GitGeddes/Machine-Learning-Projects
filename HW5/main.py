import math


# Import the training and test data sets given filenames
def import_unigram_data(data):
    temp_dict = dict()
    count = 0
    with open(data, 'rb') as textfile:
        for line in textfile:
            # Remove punctuation in a line
            sentence = remove_punct(line.split())
            for word in sentence:
                # Increment total count of all words
                count += 1
                if word not in temp_dict:
                    # Initialize word if not already in the dictionary
                    temp_dict[word] = 1
                else:
                    # Increment word frequency
                    temp_dict[word] += 1
    return temp_dict, count


# Import the training and testing data, but with pairs of words
def import_bigram_data(data):
    temp_dict = dict()
    count = 0
    with open(data, 'rb') as textfile:
        for line in textfile:
            # Don't remove punctuation in bigrams
            sentence = line.split()

            # Remove this comment for find_max() to work properly and show no punctuation
            '''sentence = remove_punct(sentence)'''

            # Only make word pairs within each sentence, not the whole document
            sentence = pair_words(sentence)
            for key in sentence:
                # Increment total count of all token pairs
                count += 1
                if key not in temp_dict:
                    # Initialize key if not already in the dictionary
                    temp_dict[key] = 1
                else:
                    # Increment pair count
                    temp_dict[key] += 1
    return temp_dict, count


# Remove punctuation from a list of tokens
def remove_punct(tokens):
    sentence = list()
    for t in tokens:
        if not is_punctuation(t[0]):
            sentence.append(t)
    return sentence


# Check if given character is a punctuation
def is_punctuation(char):
    return (char is "'" or
            char is '.' or
            char is ',' or
            char is '?' or
            char is '!' or
            char is '<')


# Loop through the four files and predict each sentence
def predict_unigram_sentences(files, dicts, counts):
    predictions = []
    # For each testing file, predict each sentence
    for i in range(len(files)):
        num_sentences = 0
        num_correct = 0
        with open(files[i], 'rb') as char:
            for line in char:
                # Remove punctuation in a line
                sentence = remove_punct(line.split())
                num_sentences += 1

                # Predict which character most likely said this sentence
                hP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[0], counts[0], sentence)
                jP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[1], counts[1], sentence)
                mP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[2], counts[2], sentence)
                rP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[3], counts[3], sentence)

                # Find the maximum probability of the four characters
                probs = [hP, jP, mP, rP]
                max_prob = max(probs)

                # Check that the max probability is the correct character
                if max_prob == probs[i]:
                    num_correct += 1

        accuracy = float(num_correct) / float(num_sentences)
        predictions.append(accuracy)
        print names[i], "accuracy:", accuracy
    return predictions


# Loop through the four files and predict each sentence, but with word pairs
def predict_bigram_sentences(files, dicts, counts):
    predictions = []
    # For each testing file, predict each sentence
    for i in range(len(files)):
        num_sentences = 0
        num_correct = 0
        with open(files[i], 'rb') as char:
            for line in char:
                # Don't remove punctuation with bigrams
                sentence = line.split()

                # Make sentence a list of pairs of words
                sentence = pair_words(sentence)

                num_sentences += 1

                # Predict which character most likely said this sentence
                hP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[0], counts[0], sentence)
                jP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[1], counts[1], sentence)
                mP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[2], counts[2], sentence)
                rP = math.log(CLASS_PROB) + calc_prob_of_sentence(dicts[3], counts[3], sentence)

                # Find the maximum probability of the four characters
                probs = [hP, jP, mP, rP]
                max_prob = max(probs)
                # print max_prob

                # Check that the max probability is the correct character
                if max_prob == probs[i]:
                    num_correct += 1

        accuracy = float(num_correct) / float(num_sentences)
        predictions.append(accuracy)
        print names[i], "accuracy:", accuracy
    return predictions


# Calculate the probabilities that a word or pair of words occurs in a data set
def calc_prob_of_sentence(dictionary, word_count, sentence):
    total = 0.0
    for key in sentence:
        if key in dictionary:
            total += math.log((dictionary[key] + PSEUDO_COUNT) / (word_count + (PSEUDO_COUNT * len(sentence))))
        else:
            total += math.log((0.0 + PSEUDO_COUNT) / (word_count + (PSEUDO_COUNT * len(sentence))))
    return total


# Take in a list of words and return a list of pairs of two consecutive words
def pair_words(sentence):
    ret_list = []
    buf = []
    for word in sentence:
        # First word in the sentence
        if len(buf) == 0:
            buf.append(word)
        # All words after the first word
        else:
            # Combine two words into a key
            ret_list.append(buf[0] + " " + word)

            # Remove old word and replace it with new word
            buf.pop()
            buf.append(word)
    return ret_list


# Find the most common key in a given list of dictionaries
# This is not required for the final project, but I kept it in
def find_max(dicts):
    for i in range(len(dicts)):
        maximum = 0
        max_key = ""
        for key in dicts[i]:
            if dicts[i][key] > maximum:
                maximum = dicts[i][key]
                max_key = key
        print names[i], "most frequent key: \t{", max_key, "\t", maximum, "}"


def main():
    # Get the training data
    train_hamlet_dict, hamlet_count = import_unigram_data(train_hamlet_filename)
    train_juliet_dict, juliet_count = import_unigram_data(train_juliet_filename)
    train_macbeth_dict, macbeth_count = import_unigram_data(train_macbeth_filename)
    train_romeo_dict, romeo_count = import_unigram_data(train_romeo_filename)

    # Predict on the unigram testing data
    test_files = [test_hamlet_filename, test_juliet_filename, test_macbeth_filename, test_romeo_filename]
    train_dicts = [train_hamlet_dict, train_juliet_dict, train_macbeth_dict, train_romeo_dict]
    word_counts = [hamlet_count, juliet_count, macbeth_count, romeo_count]
    print "Unigram data accuracy:"
    predictions = predict_unigram_sentences(test_files, train_dicts, word_counts)
    # Calculate average accuracy across the 4 characters
    total = 0.0
    for p in predictions:
        total += p
    print "Total accuracy:", (total * CLASS_PROB)

    # Find the most frequent keys of each training set
    find_max(train_dicts)

    # Get the bigram (word pairs) training data
    train_hamlet_dict, hamlet_count = import_bigram_data(train_hamlet_filename)
    train_juliet_dict, juliet_count = import_bigram_data(train_juliet_filename)
    train_macbeth_dict, macbeth_count = import_bigram_data(train_macbeth_filename)
    train_romeo_dict, romeo_count = import_bigram_data(train_romeo_filename)

    # Predict on the bigram testing data
    test_files = [test_hamlet_filename, test_juliet_filename, test_macbeth_filename, test_romeo_filename]
    train_dicts = [train_hamlet_dict, train_juliet_dict, train_macbeth_dict, train_romeo_dict]
    word_counts = [hamlet_count, juliet_count, macbeth_count, romeo_count]
    print "\nBigram data accuracy:"
    predictions = predict_bigram_sentences(test_files, train_dicts, word_counts)
    # Calculate average accuracy across the 4 characters
    total = 0.0
    for p in predictions:
        total += p
    print "Total accuracy:", (total * CLASS_PROB)

    # Find the most frequent keys of each training set
    find_max(train_dicts)


# Training file names
train_hamlet_filename = 'data/Training Files/hamlet_train.txt'
train_juliet_filename = 'data/Training Files/juliet_train.txt'
train_macbeth_filename = 'data/Training Files/macbeth_train.txt'
train_romeo_filename = 'data/Training Files/romeo_train.txt'

# Testing file names
test_hamlet_filename = 'data/Testing Files/hamlet_test.txt'
test_juliet_filename = 'data/Testing Files/juliet_test.txt'
test_macbeth_filename = 'data/Testing Files/macbeth_test.txt'
test_romeo_filename = 'data/Testing Files/romeo_test.txt'

# Character name list
names = ["Hamlet", "Juliet", "Macbeth", "Romeo"]

# Hardcoded global variables
PSEUDO_COUNT = 0.0001
CLASS_PROB = 0.25

# Run the main function
main()
