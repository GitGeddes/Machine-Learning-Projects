import csv
import random  # '''Only used for making my own data'''
import matplotlib.pyplot as plt  # For pretty plots
import numpy as np  # For np.arrange() for pretty plots


# Import the three synthetic training files
def import_data(name1, name2, name3):
    # Import the first synthetic data file
    with open(name1, 'rb') as file1:
        reader = csv.reader(file1)
        global synthetic1
        # Convert all data to floats
        for point in reader:
            synthetic1.append([float(point[0]), float(point[1])])

    # Import the second synthetic data file
    with open(name2, 'rb') as file2:
        reader = csv.reader(file2)
        global synthetic2
        # Convert all data to floats
        for point in reader:
            synthetic2.append([float(point[0]), float(point[1])])

    # Import the third synthetic data file
    with open(name3, 'rb') as file3:
        reader = csv.reader(file3)
        global synthetic3
        # Convert all data to floats
        for point in reader:
            synthetic3.append([float(point[0]), float(point[1])])


# Initialize the lists of coefficients and alphas
def init_coefficients(order):
    # Clear the lists in case they were already defined
    coefficients[:] = []
    alphas[:] = []
    # Include the + 1 for x^0 coefficient (or y-intercept)
    for i in range(order + 1):
        # Initialize all coefficients to 1.0
        coefficients.append(0.0)
        if order == 9:
            # If order is 9, initialize all alphas to 1/(2^15)
            alphas.append(1.0 / pow(2, 15))
        else:
            # Otherwise, initialize all alphas to 1/(2^order)
            alphas.append(1.0 / pow(2, order))


# Predict the y-value using the polynomial
def predict(x, coefs):
    y = 0.0
    for index in range(len(coefs)):
        # Theta_sub_i
        y += coefs[index] * pow(x, index)
    return y


# Calculate the error between the given point and the current prediction polynomial
def calc_error_on_point(point, coefs):
    # Point is a list with 2 values: x and y
    # Y is the expected prediction, the polynomial makes the prediction
    error = pow(point[1] - predict(point[0], coefs), 2)
    return error


# Calculate the mean squared error between all points and the current prediction polynomial
def mean_squared_error_on_data_set(data, coefs):
    error = 0.0
    for point in data:
        error += calc_error_on_point(point, coefs)
    return error / len(data)


# Make one iteration of gradient descent
def gradient_descent(data):
    temp_coefs = coefficients
    gradients = list()
    for i in range(len(temp_coefs)):
        ith_gradient = 0.0
        for point in data:
            x = point[0]
            y = point[1]
            ith_gradient += (1.0/len(data)) * (predict(x, temp_coefs) - y) * pow(x, i)
        temp_coefs[i] = temp_coefs[i] - float(alphas[i] * ith_gradient)
        gradients.append(ith_gradient)
    # print temp_coefs
    return temp_coefs, gradients


# Train on the given data
def poly_regression(degree, data, index):
    init_coefficients(degree)
    current_error = mean_squared_error_on_data_set(data, coefficients)
    current_iteration = 0

    # First iteration
    current_iteration += 1
    temp_coefs, current_gradients = gradient_descent(data)

    # If the error isn't changing much anymore, stop training
    while abs(current_error - mean_squared_error_on_data_set(data, temp_coefs)) > 0.0001:
        # while current_error > 0.25:
        current_iteration += 1
        current_error = mean_squared_error_on_data_set(data, coefficients)
        temp_coefs, temp_gradients = gradient_descent(data)

        changed_alphas = False
        for i in range(len(alphas)):
            # For each alpha, if the gradient has flipped, divide that alpha by 2
            if sign(temp_gradients[i]) != sign(current_gradients[i]):
                alphas[i] = alphas[i] / 2
                changed_alphas = True
        if changed_alphas:
            # If any alphas have been modified, don't change the coefficients and try this iteration again
            current_iteration -= 1
            # Make sure to save the gradients for the next attempt at this iteration
            current_gradients = temp_gradients
            continue

        # No alphas changed, so update the coefficients
        global coefficients
        coefficients = temp_coefs
        current_gradients = temp_gradients
        plot_prediction(list_of_colors[index], coefficients, index)
        ax = fig.add_subplot(111)
        del ax.lines[index]
        ax.relim()
        ax.autoscale_view()
        # print alphas

        if current_iteration >= max_iterations:
            # If the max number of iterations has been reached, stop training
            break

    # Print results and plot the scatter plot and the polynomial
    print "Total iterations:", current_iteration
    print "Final error:", mean_squared_error_on_data_set(data, coefficients)
    print "Final learning rates:", alphas
    print "Final coefficients:", coefficients
    return


# Plot the polynomial
def plot_prediction(color, coefs, index):
    t = np.arange(-2., 2., 0.05)
    ax = fig.add_subplot(111)
    list_of_plots[index], = ax.plot(t, predict(t, coefs), color)
    plt.legend(legend_list[0:index + 2])
    fig.canvas.draw()
    return


# Plot the scatter plot
def plot_scatter(data):
    x_vals = list()
    y_vals = list()
    for point in data:
        x_vals.append(point[0])
        y_vals.append(point[1])
    plt.plot(x_vals, y_vals, 'ro')
    return


# Return the sign of a number
def sign(num):
    return int(num/abs(num))


def main():
    plot_scatter(test)
    plt.title('Test')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.ion()
    plt.autoscale(True, 'y', True)
    plt.show()
    '''for i in range(len(list_of_degrees)):
        print "\nPolynomial regression on synthetic data set 1 with degree", list_of_degrees[i]
        poly_regression(list_of_degrees[i], test, i)
        plot_prediction(list_of_colors[i], coefficients, i)'''
    poly_regression(6, test, 1)
    plot_prediction(list_of_colors[1], coefficients, 0)
    plt.legend(legend_list)
    plt.show(block=True)

    '''import_data(filename1, filename2, filename3)
    
    plot_scatter(synthetic1)
    plt.title('Synthetic-1')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.ion()
    plt.show()
    for i in range(len(list_of_degrees)):
        print "\nPolynomial regression on synthetic data set 1 with degree", list_of_degrees[i]
        poly_regression(list_of_degrees[i], synthetic1, i)
        plot_prediction(list_of_colors[i], coefficients, i)
    plt.legend(legend_list)
    plt.show(block=True)

    global fig, list_of_plots
    fig = plt.figure()
    list_of_plots = [None, None, None, None]
    plt.clf()
    plot_scatter(synthetic2)
    plt.title('Synthetic-2')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.ion()
    for i in range(len(list_of_degrees)):
        print "\nPolynomial regression on synthetic data set 2 with degree", list_of_degrees[i]
        plot_prediction(list_of_colors[i], coefficients, i)
        poly_regression(list_of_degrees[i], synthetic2, i)
    plt.show(block=True)

    global fig, list_of_plots
    fig = plt.figure()
    list_of_plots = [None, None, None, None]
    plt.clf()
    plot_scatter(synthetic3)
    plt.title('Synthetic-3')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.ion()
    for i in range(len(list_of_degrees)):
        print "\nPolynomial regression on synthetic data set 3 with degree", list_of_degrees[i]
        poly_regression(list_of_degrees[i], synthetic3, i)
        plot_prediction(list_of_colors[i], coefficients, i)
    plt.show(block=True)'''


# File names and lists of the synthetic data
filename1 = 'data/synthetic-1.csv'
filename2 = 'data/synthetic-2.csv'
filename3 = 'data/synthetic-3.csv'
synthetic1 = list()
synthetic2 = list()
synthetic3 = list()

coefficients = list()
alphas = list()  # Learning rates
max_iterations = 1000
list_of_degrees = [1, 2, 4, 9]  # Degrees of polynomials to try
list_of_colors = ['red', 'green', 'blue', 'cyan']  # List of colors for the plots of polynomials

# Plot figures and lines
# plt.ion()
fig = plt.figure()
list_of_plots = [None, None, None, None]
legend_list = ['Data', 'Degree 1', 'Degree 2', 'Degree 4', 'Degree 9']


def test_poly(x):
    # y = -2 * x + 2 * x**2 + 6 * x**3 - x**4 - 2 * x**5
    y = 0.07*x**6 - 0.1*x**5 - x**4 + x**3 + 3*x**2 + x
    return y  # + (random.randrange(-50.0, 50.0)/100.0)


def make_points(length):
    temp_list = list()
    left = -2.0
    right = 2.0
    delta = (right - left) / length
    x_list = np.arange(left, right, delta)
    for point in range(length):
        # temp_list.append([x_list[point] + (random.randrange(-25.0, 25.0)/100.0), test_poly(x_list[point])])
        temp_list.append([x_list[point], test_poly(x_list[point])])
    return temp_list


size = 200
test = make_points(size)

main()
