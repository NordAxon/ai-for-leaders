############################################
# Simple Linear Regression - Own algorithm #
#HEEEEEE#
############################################
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import RegressionDatasets as regdata

# Execute/Evaluate an algorithm using a train/test split
def execute_algorithm(dataset):
    ### Scatter real data in red color ###
    data_to_plot = np.array(dataset)
    X, Y = data_to_plot.T
    plt.scatter(X, Y, color='red',s=50)

    ### Calculate regression and plot the line
    predicted = simple_linear_regression(dataset)
    plt.plot(X, predicted, color = 'blue')

    actual = [row[-1] for row in dataset]

    ### Calculate Mean Square Error
    mse = mse_metric(actual, predicted)
    return mse

def simple_linear_regression (train):
    predictions = list()
    m, k = calculate_cooficients(train)
    for row in train:
        # y = m + kx
        y_predicted = m +(k*row[0])
        predictions.append(round(y_predicted, 2))
        #predictions.append(y_predicted)
    return predictions

def calculate_cooficients(dataset):
    X = [row[0] for row in dataset]
    Y = [row[1] for row in dataset]

    x_mean = mean(X)
    y_mean = mean(Y)

    k = k_calculation(X, Y, x_mean, y_mean)
    m = m_calculation(y_mean, x_mean, k)

   #print('m, k =', m,k)

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
    # Inform about and import the dataset
    regdata.dataset_choise_print()
    dataset = regdata.dataset_choise('(Own model - No split)')

    for i in range(len(dataset[0])):
        regdata.str_column_to_float(dataset, i)

    mse = execute_algorithm(dataset)

    print('MSE: %.0f' % mse)
    print('RMSE: %.2f' % sqrt(mse))

    plt.show()


main()
