############################################
# Simple Linear Regression - Own algorithm #
############################################
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import RegressionDatasets as regdata
from random import randrange

# Execute/Evaluate an algorithm using a train/test split
def execute_algorithm(dataset, split):
    test_set = list()
    train, test = train_test_split(dataset, split)

    ### Scatter train data in - green color ###
    train_data_to_plot = np.array(train)
    X_train, Y_train = train_data_to_plot.T
    plt.scatter(X_train, Y_train, color='green', s=50)

    ### Scatter test data in - red color ###
    test_data_to_plot = np.array(test)
    X_test, Y_test = test_data_to_plot.T
    plt.scatter(X_test, Y_test, color='red', s=50)

    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    actual_test_values = [row[-1] for row in test]
    predicted_test_values = simple_linear_regression(train, test_set)

    ### Plot the regression line based on test data - blue color ###
    plt.plot(X_test, predicted_test_values, color='blue')

    ### Calculate Mean Square Error
    mse = mse_metric(actual_test_values, predicted_test_values)
    return mse

def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)

    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))

    return train, dataset_copy

def simple_linear_regression (train, test):
    predictions = list()
    m, k = calculate_cooficients(train)
    for row in test:
        # y = m + kx
        y_predicted = m +(k*row[0])
        predictions.append(round(y_predicted, 2))

    return predictions

def calculate_cooficients(train):
    X = [row[0] for row in train]
    Y = [row[1] for row in train]

    x_mean = mean(X)
    y_mean = mean(Y)

    k = k_calculation(X, Y, x_mean, y_mean)
    m = m_calculation(y_mean, x_mean, k)

    return m, k

# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))

# Calculate k = Sum((x(i)-x_mean)*(y(i)-y_mean)) / Sum((x(i)-x_mean)^2) = A / B
def k_calculation(x_values, y_values, x_mean, y_mean):
    A = 0.0
    for i in range(len(x_values)):
        A += (x_values[i] - x_mean) * (y_values[i] - y_mean)
    B = sum([(x - x_mean) ** 2 for x in x_values])
    return  A/B

# Calculate m = y_mean - k*x_mean
def m_calculation(y_mean, x_mean, k):
    return (y_mean - (k*x_mean))

# Calculate MSE Mean Squared Error = SumN(Predicted(i)-Actual(i))^2 / N
def mse_metric(actual, predicted):
    sum_error = 0.0

    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error = sum_error + (prediction_error ** 2)
    mean_squared_error = sum_error / float(len(actual))
    return mean_squared_error

def main():
    # User info & import the dataset
    regdata.dataset_choise_print()
    dataset = regdata.dataset_choise('(Own model - Split)')

    for i in range(len(dataset[0])):
        regdata.str_column_to_float(dataset, i)

    split = 0.5
    mse = execute_algorithm(dataset, split)

    print('MSE: %.0f' % mse)
    print('RMSE: %.2f' % sqrt(mse))

    plt.show()

main()
