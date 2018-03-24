import csv
import random
import numpy as np
from tqdm import tqdm


# Import the lists of image data
def import_data(data, test):
    # Import the training data
    with open(data, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        # global training_data, training_expected
        for image in reader:
            # Convert all data to integers
            for i in range(len(image)):
                image[i] = int(image[i])

            # The first element is the expected output
            training_expected.append(image[0])
            del image[0]

            # Add current image to the list of data
            training_data.append(image)

    # Import the testing data
    with open(test, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for image in reader:
            # Convert all data to integers
            for i in range(len(image)):
                image[i] = int(image[i])

            # The first element is the expected output
            testing_expected.append(image[0])
            del image[0]

            # Add current image to the list of data
            testing_data.append(image)


# Initialize the perceptron's weights
def init_network():
    # Loop between index 0 and len(node_count_list) - 1
    for i_layer in range(len(node_count_list) - 1):
        # Create the weights for the bias nodes
        bias_weights[i_layer] = random.uniform(-1.0, 1.0)
        # Create a 2D list
        weights.append(list(list()))
        # Loop between index 0 and the number of nodes in the current layer
        for i_node in range(node_count_list[i_layer]):
            # Create a list
            weights[i_layer].append(list())
            # Loop between 0 and the number of nodes in the next layer
            for i_weight in range(node_count_list[i_layer + 1]):
                # Assign random float between -1.0 and 1.0
                weights[i_layer][i_node].append(random.uniform(-1.0, 1.0))


# Count the number of correct predictions
def count_correct(predictions, expecteds):
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == expecteds[i]:
            count += 1
    return count


# Make a prediction based on the image data
def predict(data):
    outputs = list()
    # For each image, predict the output
    for i_image in tqdm(range(len(data))):
        output = 0.0
        # Calculate the output of each hidden node, and add it to the final prediction
        for i_node in range(node_count_list[1]):
            # Temporarily add the bias node and weight to the training data and weights lists
            training_data[i_image].append(1.0)
            # This needs to be a list because it's the weights from the input bias to the 3 hidden nodes
            weights[0].append([bias_weights[0], bias_weights[0], bias_weights[0]])

            # combo = weights(transposed) * inputs to this node (as a dot product)
            combo = sum([weights[0][i_input][i_node] * data[i_image][i_input] for i_input in range(len(data[i_image]))])
            activation = sigmoid(combo)

            # Add the activation of this hidden node times its weight to the rolling sum of the output
            output += weights[1][i_node][0] * activation

            # Remove the bias node and weight from the training data and weight lists
            training_data[i_image].pop()
            weights[0].pop()
        # Add the hidden layer bias
        output += bias_weights[1] * 1.0
        output = sigmoid(output)
        # Clamp outputs to either 0 or 1
        if output >= 0.5:
            output = 1
        else:
            output = 0
        outputs.append(output)
    return outputs


# Calculate the sigmoid function on the given weighted feature combination
def sigmoid(x):
    # sigmoid(in) = 1 / ( 1 + e^-in )
    # Check for overflow (30 is when the function rounds to 1.0)
    if x > 30:
        x = 30
    elif x < -30:
        x = -30
    return 1.0 / (1.0 + np.e ** (-1.0 * x))


# Run an iteration (epoch), and return percent of predictions correct
def run_epoch():
    # global weights
    outputs = list()
    # For each image, predict the output
    for i_image in tqdm(range(len(training_data))):
        output = 0.0
        activations = list()
        # Calculate the output of each hidden node, and add it to the final prediction
        for i_node in range(node_count_list[1]):
            # Temporarily add the bias node and weight to the training data and weights lists
            training_data[i_image].append(1.0)
            # This needs to be a list because it's the weights from the input bias to the 3 hidden nodes
            weights[0].append([bias_weights[0], bias_weights[0], bias_weights[0]])

            # combo = weights(transposed) * inputs to this node (as a dot product)
            combo = sum([weights[0][i_input][i_node] * training_data[i_image][i_input] for i_input in range(len(training_data[i_image]))])
            activation = sigmoid(combo)
            activations.append(activation)

            # Add the activation of this hidden node times its weight to the rolling sum of the output
            output += weights[1][i_node][0] * activation

            # Remove the bias node and weight from the training data and weight lists
            training_data[i_image].pop()
            weights[0].pop()
        # Add the hidden layer bias
        output += bias_weights[1] * 1.0
        # Calculate the output using the sigmoid function
        output = sigmoid(output)

        # Back propagation
        # Calculate delta for output node
        delta_output = (training_expected[i_image] - output) * output * (1.0 - output)
        deltas = list(list())
        hidden_deltas = list()
        # Calculate deltas for hidden layer
        for i_node in range(node_count_list[1]):
            hidden_deltas.append(activations[i_node] * (1 - activations[i_node]) * weights[1][i_node][0] * delta_output)
        # Make 2D list with list of hidden deltas first, and output delta in second list
        deltas.append(hidden_deltas)
        deltas.append([delta_output])

        # Update weights
        # Loop over input and hidden layer(s), but backwards
        for i_layer in reversed(range(len(node_count_list) - 1)):
            # Loop over all nodes in current layer
            for i_node in range(node_count_list[i_layer]):
                # Loop over all weights pointing to the next layer
                for i_weight in range(node_count_list[i_layer + 1]):
                    # Here, deltas[i_layer] corresponds to the deltas of the next forward layer
                    if i_layer == 0:
                        # Input activations for input layer are the training data
                        change = alpha * training_data[i_image][i_node] * deltas[i_layer][i_weight]
                    else:
                        change = alpha * activations[i_node] * deltas[i_layer][i_weight]
                    weights[i_layer][i_node][i_weight] = weights[i_layer][i_node][i_weight] + change
        # End back propagation

        # Clamp outputs to either 0 or 1
        if output >= 0.5:
            output = 1
        else:
            output = 0
        outputs.append(output)

    correct = count_correct(outputs, training_expected)
    percent_correct = 100.0 * correct / len(training_expected)
    print "\n", correct, "out of", len(training_expected), "correct"
    print format(percent_correct, '2.2f'), "%\n"
    return percent_correct


def run_test():
    predictions = predict(testing_data)

    correct = count_correct(predictions, testing_expected)
    percent_correct = 100.0 * correct / len(testing_expected)
    print "\n", correct, "out of", len(testing_expected), "correct"
    print format(percent_correct, '2.2f'), "%\n"
    return percent_correct


def main():
    import_data(training_filename, testing_filename)
    init_network()
    for i in range(max_epochs):
        print "Current epoch:", i + 1
        # If percent of predictions correct is over this criteria, stop
        if run_epoch() > stopping_criteria:
            print "Number of epochs taken:", i + 1
            break

    print "Running test"
    run_test()


# List of image data: Each image is a 1D-array with 784 elements (28x28 image)
training_data = list(list())
# List of expected outputs, each index corresponds to the same index in training_data
training_expected = list()
testing_data = list()
testing_expected = list()
training_filename = 'data/mnist_train_0_1.csv'
testing_filename = 'data/mnist_test_0_1.csv'

# Number of nodes per layer, with the first index being the input nodes and the
# last index being the output node
node_count_list = [784, 3, 1]
# Weights of the neural network, first list is the layer number,
# Second list is the index of the node inside the layer,
# Third list is list of weights from the current node to the next layer
weights = list(list(list()))
# Weights for the bias nodes, to be randomized later, but this shows there's only 2
bias_weights = [1, 1]

# Learning rate
alpha = 0.005
# Limit the number of iterations
max_epochs = 10
# Stop training when this percent of predictions are correct on training data
stopping_criteria = 99

# Run main
main()
