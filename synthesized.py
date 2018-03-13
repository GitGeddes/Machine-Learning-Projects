import csv
import numpy as np # unique(list), log2(num)
import matplotlib.pyplot as plt


def open_csv(name):
    # This list has 3 dimensions
    # Dimension 1 is the single bin because the data hasn't been split
    # Dimension 2 is the list of data points
    # Dimension 3 is a single data point
    ret_list = list(list(list()))
    ret_list.append(list(list()))

    # Open a .cvs file and import it into a 3D list
    with open(name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            temp_list = list(row)

            # Synthetic:
            temp_list[0] = float(temp_list[0])
            temp_list[1] = float(temp_list[1])
            temp_list[2] = int(temp_list[target_label_column])

            # Make a 3D list with 1 "bin"
            ret_list[0].append(temp_list)

        for row in ret_list[0]:
            temp_list = list(row)
            temp_list.pop(target_label_column)
            test_input.append(temp_list)

        return ret_list


# Get most common value in a list
def get_most_common(data_list):
    val = None
    for key in np.unique(data_list):
        if val is None:
            val = key
        elif data_list.count(val) < data_list.count(key):
            val = key

    return val


# Input: 2D array
# Output: entropy of this node
def node_entropy(data_list):
    entropy = 0.0
    if len(data_list) <= 0:
        entropy = 0.0
        return entropy
    else:
        label_list = [row[target_label_column] for row in data_list]

        for key in np.unique(label_list):
            prop = float(label_list.count(key)) / len(label_list)
            if prop != 0:
                entropy += (-1) * prop * np.log2(prop)
            else:
                entropy += 0
        return entropy


# Input: 3D array of a root node and child nodes
# Output: entropy of this tree
def tree_entropy(data_list):
    if len(data_list) <= 0:
        entropy = 0.0
    else:
        bin_list = list()
        for bin_ in data_list:
            for row in bin_:
                bin_list.append(row)
        entropy = node_entropy(bin_list)
    return entropy


def info_gain(data_list):
    entropy_sum = 0.0
    total = 0
    for index in range(len(data_list)):
        total += len(data_list[index])

    for bin_list in data_list:
        prop = float(len(bin_list)) / total
        entropy_sum += prop * node_entropy(bin_list)

    return tree_entropy(data_list) - entropy_sum


def categorize_continuous(data_list, att_num, num_bins):
    ret_list = list(list(list()))
    ret_list.append(list(list()))

    attribute_list = [row[att_num] for row in data_list[0]]

    max_att = max(attribute_list)
    min_att = min(attribute_list)

    if att_num == 0:
        global x_max
        x_max = max_att
        global x_min
        x_min = min_att
    else:
        global y_max
        y_max = max_att
        global y_min
        y_min = min_att

    for point in range(len(data_list[0])):
        for bin_num in range(num_bins):
            # Compare attribute value to split, accounting for floating point error
            attribute = data_list[0][point][att_num]
            split_val = ((max_att - min_att)/num_bins * (bin_num + 1)) + min_att

            if attribute - split_val < 0.000001:
                data_list[0][point][att_num] = bin_num
                ret_list[0].append(data_list[0][point])
                break

    for bin_num in range(num_bins):
        split_val = ((max_att - min_att)/num_bins * (bin_num + 1)) + min_att
        if att_num == 0:
            x_splits.append(split_val)
        else:
            y_splits.append(split_val)

    return ret_list


def make_bins_on_attribute(data_list, att_num):
    ret_list = list(list(list()))
    bin_labels = {}

    label_list = [row[att_num] for row in data_list[0]]
    num_bins = len(np.unique(label_list))
    for index in range(num_bins):
        bin_labels[index] = np.unique(label_list)[index]

    for num in range(num_bins):
        ret_list.append(list(list()))

    for point in range(len(data_list[0])):
        for bin_num in range(num_bins):
            if bin_labels[bin_num] == data_list[0][point][att_num]:
                ret_list[bin_num].append(data_list[0][point])
                break
    return ret_list


def id3(data_list, depth, attr_val):
    node = Node(depth + 1)
    if attr_val is not None:
        node.set_attribute_value(attr_val)

    # Check if entropy is less than a certain threshold,
    # or current node depth is greater than or equal to maximum
    # or number of already-split-on attributes equals number of attributes
    if node_entropy(data_list[0]) == 0.0 or depth >= max_depth or len(splits) == len(data_list[0][0]) - 1:
        # Guess based on majority label
        label_list = [row[target_label_column] for row in data_list[0]]
        node.set_leaf(get_most_common(label_list))
    else:
        t, attr = split_attribute(data_list)
        splits.append(attr)
        node.set_attribute_split(attr)
        for n in range(len(t)):
            if len(t[n]) == 0:
                # Empty node, guess
                label_list = [row[target_label_column] for row in data_list[0]]
                node.set_leaf(get_most_common(label_list))
            elif depth + 1 >= max_depth:
                # Guess before depth gets too high
                label_list = [row[target_label_column] for row in data_list[0]]
                node.set_leaf(get_most_common(label_list))
            else:
                # Recursion
                temp = list()
                temp.append(t[n])
                node.node_list.append(id3(temp, depth + 1, t[n][0][attr]))
    return node


def split_attribute(data_list):
    best_split = list(list(list()))
    best_attribute = -1
    max_info_gain = -1
    for column in range(len(data_list[0][0]) - 1):
        if not splits.__contains__(column):
            split_list = make_bins_on_attribute(data_list, column)
            gain = info_gain(split_list)
            if gain > max_info_gain:
                max_info_gain = gain
                best_split = split_list
                best_attribute = column
    return best_split, best_attribute


def categorize_user_input(user_input, attr):
    if attr == 0:
        for bin_num in range(len(x_splits)):
            if user_input[attr] - x_splits[bin_num] < 0.000001:
                # Includes any data points below range of training data
                user_input[attr] = bin_num
                break
            elif bin_num == len(y_splits) - 1:
                # Data point above range of training data, plot to last bin
                user_input[attr] = bin_num
    else:
        for bin_num in range(len(y_splits)):
            if user_input[attr] - y_splits[bin_num] < 0.000001:
                # Includes any data points below range of training data
                user_input[attr] = bin_num
                break
            elif bin_num == len(y_splits) - 1:
                # Data point above range of training data, plot to last bin
                user_input[attr] = bin_num
    return user_input


def predict(final_tree, user_input):
    if final_tree.leaf:
        # This Node is a leaf, print prediction label
        return final_tree.label
    else:
        attr = final_tree.attribute_split
        branch_value_list = list()
        for child in final_tree.node_list:
            branch_value_list.append(child.value)

        for branch_index in range(len(branch_value_list)):
            if user_input[attr] == branch_value_list[branch_index]:
                return predict(final_tree.node_list[branch_index], user_input)


def plot_predictions(data_list):
    for row in data_list:
        x_val = float(row[0])
        y_val = float(row[1])

        if int(row[target_label_column]) == 0:
            plt.scatter(x_val, y_val, color='b')
        else:
            plt.scatter(x_val, y_val, color='y')

    plt.title(file_name)
    plt.show()


class Node:
    split_list = list()
    node_list = None
    attribute_split = None
    value = None
    label = None
    leaf = False

    def __init__(self, depth):
        self.depth = depth

    def set_attribute_split(self, column):
        self.attribute_split = split_labels[column]
        self.node_list = list()

    def set_leaf(self, label):
        self.leaf = True
        self.label = label

    def set_attribute_value(self, value):
        self.value = value

    def print_node(self):
        tabs = ""
        for index in range(self.depth - 1):
            tabs += "\t"
        if self.leaf:
            print tabs, "{ Branch Value :", self.value, " |  Label :", self.label, " }"
        else:
            print tabs, "{ Branch Value :", self.value, " |  Attribute (Column) :", self.attribute_split, " }"
            if self.node_list is not None:
                for child in self.node_list:
                    child.print_node()


max_depth = 3
max_bins = 2
file_name = 'data/synthetic-4.csv'
target_label_column = 2
# List of top edge of each bin, used for categorizing user input into the bins
x_splits = list()
y_splits = list()
x_min = 0
x_max = 0
y_min = 0
y_max = 0
# List of attributes/columns we've already split on
splits = list()
split_labels = [0, 1]
# List of input rows without the label to test over
test_input = list()


def main():
    data = open_csv(file_name)
    plot_predictions(data[0])
    data = categorize_continuous(data, 0, max_bins)
    data = categorize_continuous(data, 1, max_bins)

    root = id3(data, 0, None)
    root.print_node()

    count_correct = 0
    for user in range(len(test_input)):
        test_input[user] = categorize_user_input(test_input[user], 0)
        test_input[user] = categorize_user_input(test_input[user], 1)
        if predict(root, test_input[user]) == data[0][user][target_label_column]:
            count_correct += 1

    training_error = float(len(data[0]) - count_correct) / len(data[0])
    print "Correct Predictions :", count_correct
    print "Size of data list :", len(data[0])
    print "Training Error :", training_error


main()
